#!/usr/bin/env python
"""LLM-судья v2: промпт под реальный формат данных, логит-разность, ровно top-k, режим рассуждений.

Платформа запускает:
    pip install .
    python scripts/predict.py --input <test.csv|parquet> --output <predictions.csv|parquet>

Шаги:
  1. Сигналы графа (guard.lite) — только для разрешения ничьих и запасной метки.
  2. Веса из model/ (или скачивание с Hugging Face, только для экспериментов) -> vLLM.
  3. Проход chat: один токен-ответ по всем примерам (короткие первыми).
  4. MODE == "think": второй проход с рассуждением; первыми идут примеры, ближайшие к границе
     top-k по оценке chat. Итог = z(chat) + z(think); где рассуждение не успело, z(think)
     заменяется средним по своей стороне границы.
  5. Метка 1 — ровно TOP_SHARE·N примеров с наибольшей оценкой (ничьи — по сигналам графа).
На валидации (46): chat AUC 0.915, think AUC 0.895, сумма 0.951; промпт бейзлайна — 0.819.
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
PROMPT_FILE = "prompt_v2.md"   # prompt.md — промпт бейзлайна, prompt_v2.md — под формат данных
REMINDER_ON = True             # напоминание в конце сообщения
MODE = "think"                 # "chat" | "think"
THINK_MAX = 2560               # потолок токенов рассуждения (на валидации медиана 1233, максимум 3431)
TOP_SHARE = 0.28               # доля ответов с меткой 1
CONCURRENCY = 64
TIME_BUDGET = 27 * 60          # секунд на всё, с запасом до лимита 30 минут
REQUIRE_JUDGE = True           # судья оценил меньше половины — упасть намеренно (попытка вернётся)

HF_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DOWNLOAD_DIR = ROOT / "model_hf"
DOWNLOAD_BUDGET = 15 * 60


def log(*a):
    print("[judge2]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


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


def graph_signals(prompts, responses):
    from guard.lite import check
    hot = []
    for p, r in zip(prompts, responses):
        try:
            hot.append(bool(check(str(p), str(r))["hot"]))
        except Exception:  # noqa: BLE001
            hot.append(False)
    return hot


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


def zscore(vals):
    known = [v for v in vals if v is not None]
    if len(known) < 2:
        return [None] * len(vals)
    m, sd = statistics.mean(known), statistics.pstdev(known) or 1.0
    return [None if v is None else (v - m) / sd for v in vals]


def chat_rank(scores):
    """Место примера по оценке chat (0 — самый подозрительный; без оценки — в конце)."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i] is not None, scores[i] or 0.0), reverse=True)
    rank = [0] * len(scores)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


def select_top(final, hot, share):
    """Ровно k = share·N меток; порядок — оценка, затем сигнал графа; без оценки — в конец."""
    n = len(final)
    k = int(round(share * n))
    key = lambda i: (final[i] is not None, final[i] if final[i] is not None else 0.0, hot[i])  # noqa: E731
    order = sorted(range(n), key=key, reverse=True)
    lab = [0] * n
    for i in order[:k]:
        lab[i] = 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--base-url", default=None)
    args, unknown = ap.parse_known_args()
    log(f"MODE={MODE} PROMPT={PROMPT_FILE} REMINDER={REMINDER_ON} TOP_SHARE={TOP_SHARE} лишние аргументы {unknown}")

    df = read_table(args.input)
    prompts, responses = df["prompt"].astype(str).tolist(), df["response"].astype(str).tolist()
    n = len(df)
    hot = graph_signals(prompts, responses)
    log(f"строк {n}; сигналы графа: {sum(hot)}")

    system = (ROOT / "src/hallucination_detector" / PROMPT_FILE).read_text(encoding="utf-8")
    from hallucination_detector.scoring import Judge, user_text
    texts = [user_text(p, r, REMINDER_ON) for p, r in zip(prompts, responses)]
    order = sorted(range(n), key=lambda i: len(texts[i]))          # короткие первыми
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
        chat = run_pass("chat", judge.chat, system, texts, order, T_START + TIME_BUDGET - 60, T_START + TIME_BUDGET)
        if MODE == "think" and judge.tok is not None:
            rank = chat_rank(chat)
            k = int(round(TOP_SHARE * n))
            t_order = sorted(range(n), key=lambda i: abs(rank[i] - k))   # ближайшие к границе первыми
            think = run_pass("think", judge.think, system, texts, t_order, T_START + TIME_BUDGET - 150,
                             T_START + TIME_BUDGET)
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

    zc, zt = zscore(chat), zscore(think)
    n_think = sum(v is not None for v in zt)
    if any(v is not None for v in zt):
        rank, k = chat_rank(chat), int(round(TOP_SHARE * n))
        top = [v for i, v in enumerate(zt) if v is not None and rank[i] < k]
        bot = [v for i, v in enumerate(zt) if v is not None and rank[i] >= k]
        fill_top = statistics.mean(top) if top else 0.0
        fill_bot = statistics.mean(bot) if bot else 0.0
        zt = [v if v is not None else (fill_top if rank[i] < k else fill_bot) for i, v in enumerate(zt)]
        final = [None if c is None else c + t for c, t in zip(zc, zt)]
    else:
        final = zc
    log(f"итоговых оценок: {sum(v is not None for v in final)}/{n}, из них с рассуждением {n_think}")
    labels = select_top(final, hot, TOP_SHARE)
    out = pd.DataFrame({"id": df["id"].values, "label": labels})
    write_table(out, args.output)
    log(f"записано {len(out)} -> {args.output}; помечено {sum(labels)} ({out.label.mean():.3f})")


if __name__ == "__main__":
    main()
