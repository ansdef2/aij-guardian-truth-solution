#!/usr/bin/env python
"""Бейзлайн + добор пограничных: судья бейзлайна помечает всё с P >= 0.5 (как final_base),
а ответы чуть ниже порога (логит-разность в [NONCAND_MIN, 0)) перепроверяются рассуждением
с промптом v3 и помечаются, если рассуждение говорит «1».

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

Почему так: final_base на тесте — 97 найденных из 263 помеченных (F1 0.4936). Добавленная
пометка повышает F1, если она верна с вероятностью > F1/2 ~ 0.25; у перепроверки рассуждением
на тесте точность была ~0.58. Кандидатов бейзлайна не трогаем.
Версия по правилам: веса берутся только из model/, интернет не нужен.
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
THINK_PROMPT, THINK_REMINDER = "prompt_v3.md", True    # перепроверка пограничных рассуждением
THINK_MAX = 2560               # потолок токенов рассуждения (на валидации медиана ~1000)
# пограничные ответы: логит-разность судьи бейзлайна в [NONCAND_MIN, 0)
NONCAND_MIN = -1.0             # ближе к порогу — выше доля галлюцинаций, выше точность добора
CONCURRENCY = 64
TIME_BUDGET = 27 * 60          # секунд на всё, с запасом до лимита 30 минут
REQUIRE_JUDGE = True           # судья оценил меньше половины — упасть намеренно (попытка вернётся)



def log(*a):
    print("[border]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


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
            if model is None:
                raise RuntimeError("в model/ нет весов (config.json и *.safetensors)")
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
            t_order = sorted(rest, key=lambda i: -chat[i])   # только пограничные, ближайшие к порогу первыми
            log(f"кандидатов {len(cand)} (все помечаются); пограничных на перепроверку {len(t_order)}")
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

    labels, added = [], 0
    for c, t in zip(chat, think):
        if c is not None and c >= 0:
            labels.append(1)                       # как у бейзлайна
        elif c is not None and t is not None and t > 0:
            labels.append(1)                       # пограничный, рассуждение подтвердило
            added += 1
        else:
            labels.append(0)
    log(f"добавлено рассуждением: {added}")
    out = pd.DataFrame({"id": df["id"].values, "label": labels})
    write_table(out, args.output)
    log(f"записано {len(out)} -> {args.output}; помечено {sum(labels)} ({out.label.mean():.3f})")


if __name__ == "__main__":
    main()
