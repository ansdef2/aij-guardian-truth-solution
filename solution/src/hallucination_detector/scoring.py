"""Оценка примеров судьёй: логит-разность logprob('1') − logprob('0') вместо softmax.

Softmax по двум логитам насыщается в 1.0 (если '0' выпал из top-20), и сотни примеров
получают одинаковую «уверенность». Логит-разность сохраняет порядок и без насыщения.

Два режима:
  chat  — один токен-ответ без рассуждения (как в бейзлайне);
  think — сначала рассуждение (до THINK_MAX токенов, обрывается на </think>),
          затем один токен-ответ; если рассуждение не уложилось — принудительно закрывается.
"""
from __future__ import annotations

from typing import Optional

REMINDER = ("\n\nПроверь <response> по правилам из инструкции. "
            "Есть ли в нём галлюцинация? Ответь одним символом: 1 или 0.")


def margin_from_top(top) -> Optional[float]:
    """top: [(token, logprob)] -> logprob('1') − logprob('0'); отсутствующий токен — ниже минимума top."""
    if not top:
        return None
    l1 = max((lp for t, lp in top if t.strip() == "1"), default=None)
    l0 = max((lp for t, lp in top if t.strip() == "0"), default=None)
    if l1 is None and l0 is None:
        return None
    floor = min(lp for _, lp in top) - 1.0
    return (l1 if l1 is not None else floor) - (l0 if l0 is not None else floor)


def user_text(prompt: str, response: str, reminder: bool) -> str:
    t = f"<prompt>\n{prompt}\n</prompt>\n<response>\n{response}\n</response>"
    return t + REMINDER if reminder else t


class Judge:
    def __init__(self, base_url: str, model_path: str, served: str = "judge", think_max: int = 3072, log=print):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key="EMPTY", timeout=900, max_retries=1)
        self.served, self.think_max, self.log = served, think_max, log
        self.tok = None
        self.stats = []   # (токенов рассуждения, обрезано) по каждому вызову think
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:  # noqa: BLE001
            log(f"токенизатор не загрузился ({type(e).__name__}: {e}) — режим think недоступен")

    def chat(self, system: str, user: str):
        r = self.c.chat.completions.create(
            model=self.served, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=2, logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        lp = r.choices[0].logprobs
        if not lp or not lp.content:
            return None
        for tk in lp.content:
            if tk.token.strip() in ("0", "1"):
                return margin_from_top([(t.token, t.logprob) for t in tk.top_logprobs])
        return margin_from_top([(t.token, t.logprob) for t in lp.content[0].top_logprobs])

    def think(self, system: str, user: str):
        if self.tok is None:
            return None
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        r1 = self.c.completions.create(
            model=self.served, prompt=prompt, max_tokens=self.think_max, temperature=0.6, top_p=0.95,
            stop=["</think>"], extra_body={"top_k": 20, "skip_special_tokens": False})
        ch = r1.choices[0]
        txt, trunc = ch.text, ch.finish_reason == "length"
        self.stats.append((r1.usage.completion_tokens, trunc))
        p2 = prompt + txt + ("\n</think>\n\n" if trunc or not txt.endswith("\n") else "</think>\n\n")
        r2 = self.c.completions.create(model=self.served, prompt=p2, max_tokens=1, temperature=0.0, logprobs=20,
                                       extra_body={"skip_special_tokens": False})
        lp = r2.choices[0].logprobs
        top = lp.top_logprobs[0] if lp and lp.top_logprobs else None
        return margin_from_top(list(top.items()) if top else None)
