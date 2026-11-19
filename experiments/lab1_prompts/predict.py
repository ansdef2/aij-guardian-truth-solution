#!/usr/bin/env python
"""ЛАБОРАТОРИЯ: сравнение вариантов LLM-судьи на размеченной валидации платформы (46 примеров).

Тест не размечает. Печатает метрики каждого варианта и НАМЕРЕННО завершается с кодом 3:
упавший запуск попытку не тратит, а лог становится доступен.

Варианты:
  base        — промпт бейзлайна, один токен (контроль);
  v2          — промпт под реальный формат данных и типы ошибок;
  v2_end      — v2 + короткое напоминание в конце сообщения;
  v2_think    — v2_end с режимом рассуждений (сначала рассуждение, потом один токен);
  base_think  — промпт бейзлайна с режимом рассуждений.
Оценка — логит-разность logprob('1') − logprob('0') (без насыщения в 1.0).
"""
import argparse
import math
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
DOWNLOAD_BUDGET = 15 * 60
TIME_BUDGET = 27 * 60
THINK_MAX = 4096
N_TEST = 548
REMINDER = ("\n\nПроверь <response> по правилам из инструкции. "
            "Есть ли в нём галлюцинация? Ответь одним символом: 1 или 0.")

#          имя           промпт          режим    напоминание
VARIANTS = [("base",       "prompt.md",    "chat",  False),
            ("v2",         "prompt_v2.md", "chat",  False),
            ("v2_end",     "prompt_v2.md", "chat",  True),
            ("v2_think",   "prompt_v2.md", "think", True),
            ("base_think", "prompt.md",    "think", False)]


def log(*a):
    print("[lab]", f"{time.time() - T_START:6.0f}s", *a, flush=True)


def read_table(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)


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


# ---------------------------------------------------------------- оценки
def margin_from_top(top):
    """top: [(token, logprob)] -> logprob('1') − logprob('0'); отсутствующий токен — ниже минимума top."""
    if not top:
        return None
    l1 = max((lp for t, lp in top if t.strip() == "1"), default=None)
    l0 = max((lp for t, lp in top if t.strip() == "0"), default=None)
    if l1 is None and l0 is None:
        return None
    floor = min(lp for _, lp in top) - 1.0
    return (l1 if l1 is not None else floor) - (l0 if l0 is not None else floor)


def user_text(prompt, response, reminder):
    t = f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>"
    return t + REMINDER if reminder else t


class Judge:
    def __init__(self, base_url, model_path):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key="EMPTY", timeout=900)
        self.tok = None
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_path)
            demo = self.tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
            log(f"шаблон с рассуждением, хвост: {demo[-60:]!r}")
            demo = self.tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            log(f"шаблон без рассуждения, хвост: {demo[-60:]!r}")
        except Exception as e:  # noqa: BLE001
            log(f"токенизатор не загрузился: {type(e).__name__}: {e}")

    def chat(self, system, user):
        r = self.c.chat.completions.create(
            model="judge", messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        lp = r.choices[0].logprobs
        info = {"ptok": r.usage.prompt_tokens, "ctok": 0, "trunc": False}
        if not lp or not lp.content:
            return None, info
        for tk in lp.content:
            if tk.token.strip() in ("0", "1"):
                return margin_from_top([(t.token, t.logprob) for t in tk.top_logprobs]), info
        return margin_from_top([(t.token, t.logprob) for t in lp.content[0].top_logprobs]), info

    def think(self, system, user):
        if self.tok is None:
            raise RuntimeError("нет токенизатора")
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        r1 = self.c.completions.create(
            model="judge", prompt=prompt, max_tokens=THINK_MAX, temperature=0.6, top_p=0.95, stop=["</think>"],
            extra_body={"top_k": 20, "skip_special_tokens": False})
        ch = r1.choices[0]
        txt, trunc = ch.text, ch.finish_reason == "length"
        info = {"ptok": r1.usage.prompt_tokens, "ctok": r1.usage.completion_tokens, "trunc": trunc}
        p2 = prompt + txt + ("\n</think>\n\n" if trunc or not txt.endswith("\n") else "</think>\n\n")
        r2 = self.c.completions.create(model="judge", prompt=p2, max_tokens=1, temperature=0.0, logprobs=20,
                                       extra_body={"skip_special_tokens": False})
        top = r2.choices[0].logprobs.top_logprobs[0] if r2.choices[0].logprobs else None
        return margin_from_top(list(top.items()) if top else None), info


# ---------------------------------------------------------------- метрики
def auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    win = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return win / (len(pos) * len(neg))


def f1(y, pred):
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_f1(y, s):
    return max(f1(y, [int(v >= t) for v in s]) for t in sorted(set(s)))


def zs(s):
    m, sd = statistics.mean(s), statistics.pstdev(s) or 1.0
    return [(v - m) / sd for v in s]


def report(name, y, s):
    lo = min(v for v in s if v is not None) if any(v is not None for v in s) else 0.0
    s2 = [v if v is not None else lo for v in s]
    n_none = sum(v is None for v in s)
    pos = sum(v > 0 for v in s2)
    log(f"ИТОГ {name:11s} AUC={auc(y, s2):.3f}  F1@0.5={f1(y, [int(v > 0) for v in s2]):.3f}  "
        f"bestF1={best_f1(y, s2):.3f}  помечено@0.5={pos}/{len(s2)}  без оценки={n_none}")
    return s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--port", type=int, default=8000)
    args, _ = ap.parse_known_args()

    df = read_table(args.input)
    log(f"строк {len(df)}, колонки {list(df.columns)}")
    if "label" not in df.columns:
        log("нет разметки — это не валидация; завершаюсь")
        sys.exit(3)
    y = df["label"].astype(int).tolist()
    prompts, responses = df["prompt"].astype(str).tolist(), df["response"].astype(str).tolist()

    proc = None
    results, times = {}, {}
    try:
        model = find_model() or download_model()
        if model is None:
            raise RuntimeError("нет весов")
        from hallucination_detector.server import start_vllm, wait_ready
        t0 = time.time()
        proc = start_vllm(model, "judge", args.port)
        wait_ready(args.port, timeout=15 * 60)
        log(f"vLLM готов за {time.time() - t0:.0f} с")
        judge = Judge(f"http://127.0.0.1:{args.port}/v1", model)

        for name, pfile, mode, rem in VARIANTS:
            left = TIME_BUDGET - (time.time() - T_START)
            if left < 120:
                log(f"пропуск {name}: мало времени ({left:.0f} с)")
                continue
            if mode == "think" and judge.tok is None:
                log(f"пропуск {name}: нет токенизатора")
                continue
            system = (PROMPTS / pfile).read_text(encoding="utf-8")
            errs = []
            fn = judge.chat if mode == "chat" else judge.think
            texts = [user_text(p, r, rem) for p, r in zip(prompts, responses)]

            def task(i, fn=fn, system=system, texts=texts):
                if time.time() - T_START > TIME_BUDGET:
                    return i, None, None
                try:
                    m, info = fn(system, texts[i])
                    return i, m, info
                except Exception as e:  # noqa: BLE001
                    errs.append(i)
                    if len(errs) <= 3:
                        log(f"{name}: ошибка на {i}: {type(e).__name__}: {str(e)[:300]}")
                    return i, None, None

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=64) as ex:
                out = list(ex.map(task, range(len(texts))))
            dt = time.time() - t0
            scores = [m for _, m, _ in out]
            infos = [inf for _, _, inf in out if inf]
            ptok = sum(i["ptok"] for i in infos)
            msg = f"{name}: {dt:.0f} с, ошибок {len(errs)}, prompt-токенов {ptok}, оценено {sum(s is not None for s in scores)}/{len(scores)}"
            if mode == "think" and infos:
                ct = [i["ctok"] for i in infos]
                msg += (f", рассуждение: медиана {int(statistics.median(ct))} ток., макс {max(ct)}, "
                        f"обрезано {sum(i['trunc'] for i in infos)}")
            msg += f"; грубый прогноз на тест {N_TEST}: {dt * N_TEST / len(texts) / 60:.1f} мин"
            log(msg)
            times[name] = dt
            results[name] = report(name, y, scores)
    except Exception as e:  # noqa: BLE001
        log(f"СБОЙ: {type(e).__name__}: {e}")
    finally:
        if proc is not None:
            from hallucination_detector.server import stop_vllm
            stop_vllm(proc)

    # ансамбли: среднее z-оценок
    names = list(results)
    log("---- ансамбли (среднее z-оценок) ----")
    combos = [(a, b) for k, a in enumerate(names) for b in names[k + 1:]] + ([tuple(names)] if len(names) > 2 else [])
    for c in combos:
        s = [sum(v) / len(c) for v in zip(*[zs(results[n]) for n in c])]
        log(f"ИТОГ {'+'.join(c):30s} AUC={auc(y, s):.3f} bestF1={best_f1(y, s):.3f}")

    log("---- оценки по примерам (id | метка | " + " | ".join(names) + ") ----")
    for i in range(len(df)):
        vals = " ".join(f"{results[n][i]:+.2f}" for n in names)
        print(f"ROW {df['id'].iat[i][:60]} {y[i]} {vals}", flush=True)
    log("конец лаборатории — намеренная ошибка, попытка не тратится")
    sys.exit(3)


if __name__ == "__main__":
    main()
