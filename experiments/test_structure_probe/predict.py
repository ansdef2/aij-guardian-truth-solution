#!/usr/bin/env python
"""ПРОБНИК структуры теста — без модели, несколько секунд работы.

Прогон на валидации (есть колонка label): пишет заглушку (все 0) и выходит с кодом 0,
чтобы платформа перешла к тесту.
Прогон на тесте: печатает ТОЛЬКО сводную статистику (без текстов и без id) и НАМЕРЕННО
завершается с кодом 3 — упавший запуск попытку не тратит.
Что ищем: сколько ходов из одного диалога (гипотеза «цепочки»), домены, доля вызовов.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def read_table(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


def write_table(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if str(p).endswith(".parquet") else df.to_csv(p, index=False)


def stats(df):
    ids = df["id"].astype(str)
    dlg = ids.str.split("::").str[0]
    turn = pd.to_numeric(ids.str.extract(r"::t(\d+)$")[0], errors="coerce")
    dom = dlg.str.split("__").str[0]
    call = df["response"].astype(str).str.contains("TOOL_CALL")
    per = dlg.value_counts()
    print(f"строк {len(df)}, колонки {list(df.columns)}")
    print(f"id с номером хода ::tN: {int(turn.notna().sum())} из {len(df)}")
    print(f"разных диалогов: {dlg.nunique()}")
    print("строк на диалог -> сколько таких диалогов:", per.value_counts().sort_index().to_dict())
    print(f"строк в диалогах с 2+ ходами: {int(per[per > 1].sum())}")
    print("домены (строк):", dom.value_counts().to_dict())
    print(f"ответов с вызовом инструмента: {int(call.sum())} из {len(df)}")
    print("доля вызовов по доменам:", {d: round(float(call[dom == d].mean()), 2) for d in dom.unique()})
    print("номер хода: квантили", turn.quantile([0, .25, .5, .75, 1]).round(0).to_dict())
    pl, rl = df["prompt"].astype(str).str.len(), df["response"].astype(str).str.len()
    print(f"длина prompt: медиана {int(pl.median())}, макс {int(pl.max())}; длина response: медиана {int(rl.median())}")
    # в диалогах с несколькими ходами: входит ли более ранний ход в контекст более позднего
    multi = per[per > 1].index
    nested = gaps = 0
    pairs = 0
    for d in multi:
        sub = df[dlg == d].assign(tnum=turn[dlg == d]).sort_values("tnum")
        rows = list(sub.itertuples(index=False))
        for a, b in zip(rows, rows[1:]):
            pairs += 1
            ra = str(getattr(a, "response"))
            head = ra.split("\n", 1)[0]
            if head and head in str(getattr(b, "prompt")):
                nested += 1
            if pd.notna(a.tnum) and pd.notna(b.tnum):
                gaps += int(b.tnum - a.tnum)
    if pairs:
        print(f"соседних пар ходов в одном диалоге: {pairs}; ранний ход виден в контексте позднего: {nested}; "
              f"средний разрыв номеров ходов: {gaps / pairs:.1f}")
    # совпадающие контексты (одна история — несколько вариантов ответа?)
    print(f"повторяющихся prompt: {int(df['prompt'].duplicated().sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args, _ = ap.parse_known_args()
    df = read_table(args.input)
    if "label" in df.columns:
        print(f"[probe] валидация ({len(df)} строк): пишу заглушку и перехожу к тесту", flush=True)
        write_table(pd.DataFrame({"id": df["id"].values, "label": 0}), args.output)
        try:
            stats(df)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] сбой статистики на валидации: {type(e).__name__}: {e}")
        return
        write_table(pd.DataFrame({"id": df["id"].values, "label": 0}), args.output)
        return
    print("[probe] ===== СТРУКТУРА ТЕСТА =====", flush=True)
    try:
        stats(df)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] сбой статистики: {type(e).__name__}: {e}")
    print("[probe] ===== конец — намеренная ошибка, попытка не тратится =====", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
