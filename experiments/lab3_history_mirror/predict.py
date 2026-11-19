#!/usr/bin/env python
"""ЛАБОРАТОРИЯ 3: строгая и мягкая перепроверка рассуждением на реалистичных верных ходах.

Наборы:
  REAL — 46 размеченных примеров валидации;
  HIST — прошлые ходы агента из истории тех же диалогов (до 3 на диалог): контекст до хода +
         сам ход без результатов инструментов. Почти все такие ходы верные, а по составу
         (тексты и вызовы) они похожи на тест. Доля пометок на HIST ~ доля ложных тревог на тесте.
Якоря: судья бейзлайна на тесте помечает 38% верных ответов; строгая перепроверка v3 на тесте
дала precision 0.58 при 52 найденных.
Тест не размечает: печатает метрики и НАМЕРЕННО завершается с кодом 3 (попытка не тратится).
"""
import argparse
import os
import random
import re
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
TIME_BUDGET = 25 * 60
THINK_MAX = 2560
HIST_PER_DIALOG = 3
HIST_MAX = 140

#          имя         промпт          режим    напоминание
VARIANTS = [("base",     "prompt.md",    "chat",  False),
            ("v4_think", "prompt_v4.md", "think", True),
            ("v3_think", "prompt_v3.md", "think", True)]

TURN_RE = re.compile(r"^⟦ASSISTANT · ход (\d+)⟧[ \t]*$", re.M)


def log(*a):
    print("[lab3]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


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


def hist_turns(prompt):
    """Прошлые ходы агента: (контекст до хода, ход без строк ← TOOL_RESPONSE, номер хода)."""
    out = []
    for m in TURN_RE.finditer(prompt):
        nxt = re.search(r"^⟦", prompt[m.end():], re.M)
        end = m.end() + nxt.start() if nxt else len(prompt)
        block = prompt[m.start():end]
        resp = "\n".join(l for l in block.split("\n") if not l.lstrip().startswith("←")).strip()
        if len(resp) - len(m.group(0)) < 15:
            continue
        out.append((prompt[:m.start()].rstrip(), resp, int(m.group(1))))
    return out


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

    # примеры: (id, prompt, response, label, набор, тип)
    ex = []
    for _, r in df.iterrows():
        resp = str(r["response"])
        ex.append((str(r["id"]), str(r["prompt"]), resp, int(r["label"]), "REAL",
                   "call" if "TOOL_CALL" in resp else "text"))
    rng, seen, hist = random.Random(7), set(), []
    for _, r in df.iterrows():
        dlg = str(r["id"]).split("::")[0]
        if dlg in seen:
            continue
        seen.add(dlg)
        turns = hist_turns(str(r["prompt"]))
        for ctx, resp, t in rng.sample(turns, min(HIST_PER_DIALOG, len(turns))):
            hist.append((f"HIST:{dlg}::t{t}", ctx, resp, 0, "HIST", "call" if "TOOL_CALL" in resp else "text"))
    rng.shuffle(hist)
    ex += hist[:HIST_MAX]
    n = len(ex)
    nh = sum(e[4] == "HIST" for e in ex)
    log(f"примеров: {n} (REAL {len(df)}, HIST {nh}: текстов {sum(e[4] == 'HIST' and e[5] == 'text' for e in ex)}, "
        f"вызовов {sum(e[4] == 'HIST' and e[5] == 'call' for e in ex)})")

    from hallucination_detector.scoring import Judge, user_text
    proc, results = None, {}
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
            if left < 360:
                log(f"пропуск {name}: мало времени ({left:.0f} с)")
                continue
            system = (PROMPTS / pfile).read_text(encoding="utf-8")
            fn = judge.chat if mode == "chat" else judge.think
            texts = [user_text(e[1], e[2], rem) for e in ex]
            judge.stats, errs = [], []

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
            results[name] = scores
            report(name, ex, scores)
    except Exception as e:  # noqa: BLE001
        log(f"СБОЙ: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    names = list(results)
    log("---- оценки по примерам (id | метка | набор | тип | " + " | ".join(names) + ") ----")
    for i, e in enumerate(ex):
        vals = " ".join("nan" if results[k][i] is None else f"{results[k][i]:+.2f}" for k in names)
        print(f"ROW {e[0][:70]} {e[3]} {e[4]} {e[5]} {vals}", flush=True)
    log("конец лаборатории 3 — намеренная ошибка, попытка не тратится")
    sys.exit(3)


def report(name, ex, s):
    def flagged(i):
        return s[i] is not None and s[i] >= 0

    real = [i for i, e in enumerate(ex) if e[4] == "REAL"]
    tp = sum(1 for i in real if ex[i][3] == 1 and flagged(i))
    fp = sum(1 for i in real if ex[i][3] == 0 and flagged(i))
    yr = [ex[i][3] for i in real]
    sr = [s[i] if s[i] is not None else -99 for i in real]
    parts = []
    for typ in ("text", "call", None):
        idx = [i for i, e in enumerate(ex) if e[4] == "HIST" and (typ is None or e[5] == typ)]
        if idx:
            k = sum(flagged(i) for i in idx)
            parts.append(f"{typ or 'все'} {k}/{len(idx)} ({k / len(idx):.0%})")
    log(f"ИТОГ {name:9s} REAL: AUC={auc(yr, sr):.3f} TP={tp}/23 FP={fp}/23 | HIST помечено: " + ", ".join(parts))


if __name__ == "__main__":
    main()
