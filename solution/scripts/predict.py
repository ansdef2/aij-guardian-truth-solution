#!/usr/bin/env python
"""Финальная версия: LLM-судья в режиме одного токена, веса лежат в model/.

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

  1. vLLM поднимает Qwen3.6-35B-A3B-FP8 из папки model/ (интернет не нужен).
  2. Каждый пример оценивается одним токеном: логит-разность logprob('1') − logprob('0').
  3. Метка 1, если разность >= 0 (то же, что вероятность галлюцинации >= 0.5).
Защита: короткие диалоги идут подряд (кэш префиксов), после TIME_BUDGET новые запросы
не отправляются, файл с ответами пишется в любом случае.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

T_START = time.time()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROMPT_FILE = "prompt.md"
CONCURRENCY = 64
TIME_BUDGET = 27 * 60          # секунд на всё, с запасом до лимита 30 минут
REQUIRE_JUDGE = True           # судья оценил меньше половины — упасть (запуск не засчитается)


def log(*a):
    print("[final]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


def read_table(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def write_table(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if str(p).endswith(".parquet") else df.to_csv(p, index=False)


def find_model():
    m = ROOT / "model"
    if (m / "config.json").exists() and any(m.glob("*.safetensors")):
        return str(m)
    return None


def dialog_key(id_):
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

    df = read_table(args.input)
    n = len(df)
    log(f"строк {n}; лишние аргументы {unknown}")
    from hallucination_detector.scoring import Judge, user_text
    system = (ROOT / "src/hallucination_detector" / PROMPT_FILE).read_text(encoding="utf-8")
    texts = [user_text(p, r, False) for p, r in zip(df["prompt"].astype(str), df["response"].astype(str))]
    ids = df["id"].tolist()
    order = sorted(range(n), key=lambda i: dialog_key(ids[i]))
    scores = [None] * n

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
        judge = Judge(base, model or "", log=log)

        def task(i):
            if time.time() - T_START > TIME_BUDGET - 60:
                return i, None
            try:
                return i, judge.chat(system, texts[i])
            except Exception as e:  # noqa: BLE001
                log(f"ошибка на {i}: {type(e).__name__}: {str(e)[:200]}")
                return i, None

        ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
        futs = [ex.submit(task, i) for i in order]
        done, pending = wait(futs, timeout=max(1.0, T_START + TIME_BUDGET - time.time()))
        ex.shutdown(wait=False, cancel_futures=True)
        for f in done:
            i, s = f.result()
            scores[i] = s
        log(f"оценено {sum(s is not None for s in scores)}/{n}, брошено {len(pending)}")
    except Exception as e:  # noqa: BLE001 — при любой беде всё равно пишем файл
        log(f"судья недоступен: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    got = sum(s is not None for s in scores)
    if REQUIRE_JUDGE and got < 0.5 * n:
        log(f"судья оценил {got} из {n} — завершаюсь с ошибкой")
        sys.exit(4)
    labels = [int(s is not None and s >= 0) for s in scores]
    out = pd.DataFrame({"id": df["id"].values, "label": labels})
    write_table(out, args.output)
    log(f"записано {len(out)} -> {args.output}; помечено {sum(labels)} ({out.label.mean():.3f})")


if __name__ == "__main__":
    main()
