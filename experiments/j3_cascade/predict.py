#!/usr/bin/env python
"""Каскад: кандидаты судьи бейзлайна -> перепроверка рассуждением.

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

Шаги:
  1. Судья бейзлайна (prompt.md, один токен) по всем примерам. Кандидаты — оценка > 0
     (вероятность > 0.5). На тесте это ~254 ответа, среди них 94 из 130 галлюцинаций.
  2. Перепроверка кандидатов рассуждением с промптом v3 (проверка каждого факта в сообщении
     и каждого вызова). На валидации у перепроверки 0 ложных тревог на 23 верных ответах.
     Оставшееся время — на некандидатов с оценкой чуть ниже порога (ловим пропуски шага 1).
  3. Метка = вердикт рассуждения. Где рассуждение не успело: кандидат с оценкой не ниже
     медианы кандидатов -> 1, остальные -> 0.
Если весов нет в model/, они скачиваются с Hugging Face (только для экспериментов).
"""
import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

T_START = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------- настройки
CHAT_PROMPT, CHAT_REMINDER = "prompt.md", False        # генератор кандидатов — судья бейзлайна
THINK_PROMPT, THINK_REMINDER = "prompt_v3.md", True    # перепроверка рассуждением
THINK_MAX = 2560               # потолок токенов рассуждения (на валидации медиана ~1000)
THINK_NONCAND = True           # оставшееся время — на некандидатов с оценкой не ниже NONCAND_MIN
NONCAND_MIN = -1.5             # логит-разность; на валидации большинство пропусков бейзлайна лежит в [-1.4, 0)
CONCURRENCY = 64
TIME_BUDGET = 27 * 60          # секунд на всё, с запасом до лимита 30 минут
REQUIRE_JUDGE = True           # судья оценил меньше половины — упасть намеренно (попытка вернётся)

HF_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DOWNLOAD_DIR = ROOT / "model_hf"
DOWNLOAD_BUDGET = 15 * 60


def log(*a):
    print("[judge3]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--base-url", default=None)
    args, unknown = ap.parse_known_args()
    log(f"каскад {CHAT_PROMPT} -> {THINK_PROMPT}; лишние аргументы {unknown}")

    df = read_table(args.input)
    prompts, responses = df["prompt"].astype(str).tolist(), df["response"].astype(str).tolist()
    ids = df["id"].tolist()
    n = len(df)
    from hallucination_detector.scoring import Judge, user_text
    sys_chat = (ROOT / "src/hallucination_detector" / CHAT_PROMPT).read_text(encoding="utf-8")
    sys_think = (ROOT / "src/hallucination_detector" / THINK_PROMPT).read_text(encoding="utf-8")
    t_chat = [user_text(p, r, CHAT_REMINDER) for p, r in zip(prompts, responses)]
    t_think = [user_text(p, r, THINK_REMINDER) for p, r in zip(prompts, responses)]
    order = sorted(range(n), key=lambda i: dialog_key(ids[i]))
    chat, think = [None] * n, [None] * n

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
        chat = run_pass("chat", judge.chat, sys_chat, t_chat, order,
                        T_START + TIME_BUDGET - 60, T_START + TIME_BUDGET)
        if judge.tok is not None:
            # кандидаты — как у бейзлайна: P >= 0.5 <=> логит-разность >= 0
            cand = [i for i in range(n) if chat[i] is not None and chat[i] >= 0]
            rest = [i for i in range(n) if chat[i] is not None and NONCAND_MIN <= chat[i] < 0]
            t_order = sorted(cand, key=lambda i: -chat[i])
            if THINK_NONCAND:
                t_order += sorted(rest, key=lambda i: -chat[i])
            log(f"кандидатов {len(cand)}; на перепроверку в очереди {len(t_order)}")
            think = run_pass("think", judge.think, sys_think, t_think, t_order,
                             T_START + TIME_BUDGET - 150, T_START + TIME_BUDGET)
    except Exception as e:  # noqa: BLE001 — при любой беде всё равно пишем файл
        log(f"судья недоступен: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    got = sum(s is not None for s in chat)
    if REQUIRE_JUDGE and got < 0.5 * n:
        log(f"судья оценил {got} из {n} — завершаюсь с ошибкой, чтобы попытка не списалась")
        sys.exit(4)

    cand_scores = [c for c in chat if c is not None and c >= 0]
    med = statistics.median(cand_scores) if cand_scores else float("inf")
    labels, src = [], {"think": 0, "fallback": 0}
    for c, t in zip(chat, think):
        if t is not None:
            labels.append(int(t > 0))
            src["think"] += 1
        else:
            labels.append(int(c is not None and c >= 0 and c >= med))
            src["fallback"] += 1
    out = pd.DataFrame({"id": df["id"].values, "label": labels})
    write_table(out, args.output)
    log(f"записано {len(out)} -> {args.output}; помечено {sum(labels)} ({out.label.mean():.3f}); "
        f"по рассуждению {src['think']}, запасная метка {src['fallback']}")


if __name__ == "__main__":
    main()
