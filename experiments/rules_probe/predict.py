#!/usr/bin/env python
"""Точка входа решения (ввод-вывод — как в бейзлайне организаторов).

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

Вход:  CSV/parquet с колонками id, prompt, response.
Выход: CSV/parquet с колонками id, label (0/1); формат выбирается по расширению --output.

MODE:
  all1  — все ответы помечаются 1 (зонд: доля галлюцинаций в тесте);
  rules — 1, если в ответе есть вызов инструмента или граф нашёл структурный сигнал
          (инструмент вне схемы, повтор упавшего вызова, сущность вне контекста, лишние ключи).
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODE = "rules"
TAG = re.compile(r"<prompt>(.*?)</prompt>\s*<response>(.*?)</response>", re.S)


def read_table(p: str) -> pd.DataFrame:
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def write_table(df: pd.DataFrame, p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if str(p).endswith(".parquet") else df.to_csv(p, index=False)


def split_text(prompt, response):
    """Если пример пришёл одним текстом с тегами <prompt>/<response> — разделить."""
    prompt, response = "" if prompt is None else str(prompt), "" if response is None else str(response)
    m = TAG.search(prompt) or TAG.search(response)
    if m:
        return m.group(1), m.group(2)
    return prompt, response


def predict_one(prompt, response) -> int:
    if MODE == "all1":
        return 1
    try:
        from guard.lite import check
        r = check(prompt, response)
        return int(r["has_call"] or r["hot"] or r["warm"])
    except Exception as e:  # noqa: BLE001 — на неожиданном формате не падаем
        print(f"[guard] fallback: {type(e).__name__}: {e}", file=sys.stderr)
        return int("TOOL_CALL" in response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args, unknown = ap.parse_known_args()
    t0 = time.time()
    print(f"[guard] mode={MODE} input={args.input} output={args.output} unknown={unknown}", flush=True)

    df = read_table(args.input)
    print(f"[guard] rows={len(df)} columns={list(df.columns)}", flush=True)
    cols = {c.lower().strip(): c for c in df.columns}
    p_col, r_col = cols.get("prompt"), cols.get("response")
    preds = []
    for i in range(len(df)):
        p = df[p_col].iat[i] if p_col else None
        r = df[r_col].iat[i] if r_col else None
        if p is None and r is None:               # один текстовый столбец с тегами
            p = " ".join(str(v) for v in df.iloc[i].values)
        p, r = split_text(p, r)
        preds.append(predict_one(p, r))

    out = pd.DataFrame({"id": df[cols.get("id", "id")].values, "label": preds})
    write_table(out, args.output)
    print(f"[guard] записано {len(out)} предсказаний -> {args.output}; доля 1 = {out.label.mean():.3f}; "
          f"{time.time() - t0:.1f} с", flush=True)


if __name__ == "__main__":
    main()
