#!/usr/bin/env python
"""ЛАБОРАТОРИЯ 2: каскад «кандидаты бейзлайна -> перепроверка рассуждением».

Наборы:
  REAL — 46 размеченных примеров валидации платформы;
  SYN  — 17 синтетических текстовых галлюцинаций (верный текстовый ответ + одна внесённая ошибка)
         против 21 исходного верного текстового ответа.
Тест не размечает: печатает метрики и НАМЕРЕННО завершается с кодом 3 (попытка не тратится).
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
PROMPTS = ROOT / "src" / "hallucination_detector"

HF_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DOWNLOAD_DIR = ROOT / "model_hf"
TIME_BUDGET = 27 * 60
THINK_MAX = 3072

#          имя           промпт          режим    напоминание
VARIANTS = [("base",       "prompt.md",    "chat",  False),
            ("v3",         "prompt_v3.md", "chat",  True),
            ("base_think", "prompt.md",    "think", False),
            ("v3_think",   "prompt_v3.md", "think", True),
            ("v2_think",   "prompt_v2.md", "think", True)]


def log(*a):
    print("[lab2]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


def find_model():
    for c in [str(ROOT / "model"), os.environ.get("MODEL_PATH", "")]:
        if c and Path(c, "config.json").exists() and any(Path(c).glob("*.safetensors")):
            return c
    return None


def download_model():
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(HF_REPO, local_dir=str(DOWNLOAD_DIR), max_workers=16)
    log(f"скачано за {time.time() - t0:.0f} с")
    return path


def auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8000)
    args, _ = ap.parse_known_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    log(f"строк {len(df)}, колонки {list(df.columns)}")
    if "label" not in df.columns:
        log("нет разметки — это не валидация; завершаюсь")
        sys.exit(3)

    from hallucination_detector.synthetic import build
    syn, problems = build(df)
    for p in problems:
        log(f"синтетика: {p}")
    # примеры: (id, prompt, response, label, набор, тип)
    ex = []
    for _, r in df.iterrows():
        is_text = "TOOL_CALL" not in str(r["response"])
        ex.append((str(r["id"]), str(r["prompt"]), str(r["response"]), int(r["label"]),
                   "REAL", "text" if is_text else "call"))
    for sid, p, resp, lab, kind in syn:
        ex.append((sid, p, resp, lab, "SYN", kind))
    n = len(ex)
    log(f"примеров: {n} (REAL {len(df)}, SYN {len(syn)})")

    from hallucination_detector.scoring import Judge, user_text
    proc, results, times = None, {}, {}
    try:
        model = find_model() or download_model()
        from hallucination_detector.server import start_vllm, wait_ready
        t0 = time.time()
        proc = start_vllm(model, "judge", args.port)
        wait_ready(args.port, timeout=15 * 60)
        log(f"vLLM готов за {time.time() - t0:.0f} с")
        judge = Judge(f"http://127.0.0.1:{args.port}/v1", model, think_max=THINK_MAX, log=log)

        for name, pfile, mode, rem in VARIANTS:
            left = TIME_BUDGET - (time.time() - T_START)
            if left < 200:
                log(f"пропуск {name}: мало времени ({left:.0f} с)")
                continue
            system = (PROMPTS / pfile).read_text(encoding="utf-8")
            fn = judge.chat if mode == "chat" else judge.think
            texts = [user_text(e[1], e[2], rem) for e in ex]
            judge.stats = []
            errs = []

            def task(i, fn=fn, system=system, texts=texts, errs=errs, name=name):
                if time.time() - T_START > TIME_BUDGET:
                    return None
                try:
                    return fn(system, texts[i])
                except Exception as e:  # noqa: BLE001
                    errs.append(i)
                    if len(errs) <= 3:
                        log(f"{name}: ошибка на {i}: {type(e).__name__}: {str(e)[:300]}")
                    return None

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=64) as pool:
                scores = list(pool.map(task, range(n)))
            dt = time.time() - t0
            msg = f"{name}: {dt:.0f} с, ошибок {len(errs)}, оценено {sum(s is not None for s in scores)}/{n}"
            if mode == "think" and judge.stats:
                ct = [c for c, _ in judge.stats]
                msg += (f", рассуждение: медиана {int(statistics.median(ct))} ток., макс {max(ct)}, "
                        f"обрезано {sum(t for _, t in judge.stats)}")
            log(msg)
            lo = min([s for s in scores if s is not None] or [0.0])
            results[name] = [s if s is not None else lo - 1 for s in scores]
            times[name] = dt
            report(name, ex, results[name])
    except Exception as e:  # noqa: BLE001
        log(f"СБОЙ: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    # каскад: кандидаты base > 0, перепроверка X_think > 0
    if "base" in results:
        log("---- каскад: кандидаты бейзлайна (оценка > 0) -> перепроверка рассуждением ----")
        cand = [s > 0 for s in results["base"]]
        cascade(ex, "только base", cand)
        for v in [k for k in results if k.endswith("_think")]:
            cascade(ex, f"base -> {v}", [c and t > 0 for c, t in zip(cand, results[v])])
            cascade(ex, f"{v} без каскада", [t > 0 for t in results[v]])

    names = list(results)
    log("---- оценки по примерам (id | метка | набор | тип | " + " | ".join(names) + ") ----")
    for i, e in enumerate(ex):
        vals = " ".join(f"{results[k][i]:+.2f}" for k in names)
        print(f"ROW {e[0][:70]} {e[3]} {e[4]} {e[5]} {vals}", flush=True)
    log("конец лаборатории 2 — намеренная ошибка, попытка не тратится")
    sys.exit(3)


def subset(ex, vals, cond):
    idx = [i for i, e in enumerate(ex) if cond(e)]
    return [ex[i][3] for i in idx], [vals[i] for i in idx]


def report(name, ex, s):
    yr, sr = subset(ex, s, lambda e: e[4] == "REAL")
    ys, ss = subset(ex, s, lambda e: e[4] == "SYN" or (e[4] == "REAL" and e[5] == "text" and e[3] == 0))
    tp = sum(1 for y, v in zip(yr, sr) if y == 1 and v > 0)
    fp = sum(1 for y, v in zip(yr, sr) if y == 0 and v > 0)
    syn_hit = sum(1 for y, v in zip(ys, ss) if y == 1 and v > 0)
    txt_fp = sum(1 for y, v in zip(ys, ss) if y == 0 and v > 0)
    log(f"ИТОГ {name:11s} REAL: AUC={auc(yr, sr):.3f} TP={tp}/23 FP={fp}/23 | "
        f"SYN: AUC={auc(ys, ss):.3f} поймано {syn_hit}/{sum(ys)}, ложных на верных текстах {txt_fp}/{len(ys) - sum(ys)}")


def cascade(ex, name, flag):
    real = [(e[3], f) for e, f in zip(ex, flag) if e[4] == "REAL"]
    syn = [(e[3], f) for e, f in zip(ex, flag) if e[4] == "SYN"]
    txt0 = [f for e, f in zip(ex, flag) if e[4] == "REAL" and e[5] == "text" and e[3] == 0]
    tp = sum(1 for y, f in real if y == 1 and f)
    fp = sum(1 for y, f in real if y == 0 and f)
    log(f"КАСКАД {name:26s} REAL TP={tp}/23 FP={fp}/23 | SYN поймано {sum(f for _, f in syn)}/{len(syn)} | "
        f"ложных на верных текстах {sum(txt0)}/{len(txt0)}")


if __name__ == "__main__":
    main()
