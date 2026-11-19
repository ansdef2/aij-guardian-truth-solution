#!/usr/bin/env python
"""Разделение по типу ответа: тексты — судья бейзлайна, вызовы инструментов — рассуждение.

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

Гипотеза: ложные тревоги бейзлайна (160 на тесте) сосредоточены на верных вызовах
инструментов — на валидации он пометил 2 из 2 верных вызовов и лишь 3 из 21 верного текста,
а в тесте верных вызовов ~140. Перепроверка всех кандидатов (каскад) выкинула и настоящие
текстовые галлюцинации: 52 найдено из 90 помеченных.

  1. Ответы без TOOL_CALL — вердикт судьи бейзлайна (prompt.md, один токен, P >= 0.5).
  2. Ответы с TOOL_CALL — вердикт рассуждения с промптом v3 (на валидации 14 из 21
     ошибочного вызова найдено, 0 ложных тревог). Где рассуждение не успело — метка 0.
  3. Если режим рассуждений недоступен — вызовы тоже по вердикту бейзлайна.
Если весов нет в model/, они скачиваются с Hugging Face (только для экспериментов).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

T_START = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------- настройки
CHAT_PROMPT, CHAT_REMINDER = "prompt.md", False        # тексты — судья бейзлайна
THINK_PROMPT, THINK_REMINDER = "prompt_v3.md", True    # вызовы — рассуждение
THINK_MAX = 2560               # потолок токенов рассуждения (на валидации медиана ~1000)
CONCURRENCY = 64
TIME_BUDGET = 27 * 60          # секунд на всё, с запасом до лимита 30 минут
REQUIRE_JUDGE = True           # судья оценил меньше половины — упасть намеренно (попытка вернётся)

HF_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DOWNLOAD_DIR = ROOT / "model_hf"
DOWNLOAD_BUDGET = 15 * 60


def log(*a):
    print("[judge4]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


def read_table(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def write_table(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if str(p).endswith(".parquet") else df.to_csv(p, index=False)


def find_model():
    for c in [str(ROOT / "model"), os.environ.get("MODEL_PATH", "")]:
        if c and Path(c, "config.json").exists() and any(Path(c).glob("*.safetensors")):
            return c
    return None


def download_model():
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from concurrent.futures import ThreadPoolExecutor as TPE, TimeoutError as TE
    from huggingface_hub import snapshot_download
    t0 = time.time()
    with TPE(1) as ex:
        fut = ex.submit(snapshot_download, HF_REPO, local_dir=str(DOWNLOAD_DIR), max_workers=16)
        try:
            path = fut.result(timeout=DOWNLOAD_BUDGET)
        except TE:
            log("скачивание не уложилось в бюджет")
            return None
    log(f"скачано за {time.time() - t0:.0f} с")
    return path


def run_pass(name, fn, system, texts, order, deadline, hard_deadline):
    """Новые запросы не начинаются после deadline; после hard_deadline недождавшиеся отбрасываются."""
    from concurrent.futures import wait
    scores = [None] * len(texts)

    def task(i):
        if time.time() > deadline:
            return i, None, "время"
        try:
            return i, fn(system, texts[i]), None
        except Exception as e:  # noqa: BLE001
            return i, None, f"{type(e).__name__}: {str(e)[:200]}"

    t0, errs = time.time(), []
    ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
    futs = [ex.submit(task, i) for i in order]
    done, pending = wait(futs, timeout=max(1.0, hard_deadline - time.time()))
    ex.shutdown(wait=False, cancel_futures=True)
    for f in done:
        i, s, err = f.result()
        scores[i] = s
        if err and err != "время":
            errs.append(err)
            if len(errs) <= 3:
                log(f"{name}: ошибка на {i}: {err}")
    got = sum(s is not None for s in scores)
    log(f"{name}: {time.time() - t0:.0f} с, оценено {got}/{len(texts)}, ошибок {len(errs)}, брошено {len(pending)}")
    return scores


def dialog_key(id_):
    """Группируем ходы одного диалога подряд — так лучше работает кэш префиксов vLLM."""
    d, _, t = str(id_).partition("::t")
    return d, int(t) if t.isdigit() else 0


def is_call(response):
    return "TOOL_CALL" in str(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--base-url", default=None)
    args, unknown = ap.parse_known_args()
    log(f"тексты: {CHAT_PROMPT}; вызовы: {THINK_PROMPT} с рассуждением; лишние аргументы {unknown}")

    df = read_table(args.input)
    prompts, responses = df["prompt"].astype(str).tolist(), df["response"].astype(str).tolist()
    ids = df["id"].tolist()
    n = len(df)
    call = [is_call(r) for r in responses]
    order = sorted(range(n), key=lambda i: dialog_key(ids[i]))
    text_idx = [i for i in order if not call[i]]
    call_idx = [i for i in order if call[i]]
    log(f"строк {n}: текстов {len(text_idx)}, вызовов {len(call_idx)}")

    from hallucination_detector.scoring import Judge, user_text
    sys_chat = (ROOT / "src/hallucination_detector" / CHAT_PROMPT).read_text(encoding="utf-8")
    sys_think = (ROOT / "src/hallucination_detector" / THINK_PROMPT).read_text(encoding="utf-8")
    t_chat = [user_text(p, r, CHAT_REMINDER) for p, r in zip(prompts, responses)]
    t_think = [user_text(p, r, THINK_REMINDER) for p, r in zip(prompts, responses)]
    chat, think = [None] * n, [None] * n
    think_ok = False

    proc, base = None, args.base_url
    try:
        model = find_model()
        if base is None:
            model = model or download_model()
            if model is None:
                raise RuntimeError("веса модели не найдены и не скачались")
            from hallucination_detector.server import start_vllm, wait_ready
            proc = start_vllm(model, "judge", args.port, tp=args.tp)
            wait_ready(args.port, timeout=max(60, TIME_BUDGET - (time.time() - T_START) - 120))
            base = f"http://127.0.0.1:{args.port}/v1"
            log("vLLM готов")
        judge = Judge(base, model or "", think_max=THINK_MAX, log=log)
        think_ok = judge.tok is not None
        # тексты — всегда судьёй бейзлайна; вызовы — им же, только если рассуждения недоступны
        chat_idx = text_idx if think_ok else order
        chat = run_pass("chat", judge.chat, sys_chat, t_chat, chat_idx,
                        T_START + TIME_BUDGET - 60, T_START + TIME_BUDGET)
        if think_ok:
            think = run_pass("think", judge.think, sys_think, t_think, call_idx,
                             T_START + TIME_BUDGET - 150, T_START + TIME_BUDGET)
    except Exception as e:  # noqa: BLE001 — при любой беде всё равно пишем файл
        log(f"судья недоступен: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    got = sum(chat[i] is not None for i in text_idx)
    if REQUIRE_JUDGE and got < 0.5 * max(len(text_idx), 1):
        log(f"судья оценил {got} из {len(text_idx)} текстов — завершаюсь с ошибкой, чтобы попытка не списалась")
        sys.exit(4)

    labels, src = [], {"текст": 0, "вызов-рассуждение": 0, "вызов-бейзлайн": 0, "без оценки": 0}
    for i in range(n):
        if not call[i]:
            labels.append(int(chat[i] is not None and chat[i] >= 0))
            src["текст"] += 1
        elif think[i] is not None:
            labels.append(int(think[i] > 0))
            src["вызов-рассуждение"] += 1
        elif not think_ok and chat[i] is not None:
            labels.append(int(chat[i] >= 0))
            src["вызов-бейзлайн"] += 1
        else:
            labels.append(0)
            src["без оценки"] += 1
    out = pd.DataFrame({"id": df["id"].values, "label": labels})
    write_table(out, args.output)
    log(f"записано {len(out)} -> {args.output}; помечено {sum(labels)} ({out.label.mean():.3f}); {src}")


if __name__ == "__main__":
    main()
