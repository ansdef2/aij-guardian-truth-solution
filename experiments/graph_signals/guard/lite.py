"""Облегчённая проверка графа: только стандартная библиотека Python.

Считает структурные сигналы, которые в полной методике дают температуру ≥ 30°:
  tool    — вызванный инструмент отсутствует в [AVAILABLE TOOLS];
  loop    — повтор вызова с теми же аргументами, который уже вернул ошибку;
  entity  — сущность ответа (ID, дата, число, значение аргумента) не найдена в контексте;
  schema  — ключ аргумента не описан в схеме инструмента.
Лексика и синтаксис (Natasha) здесь не нужны: эти сигналы на них не опираются.
"""
from __future__ import annotations

import json
import re

from .entities import extract_entities, norm_num
from .parse import parse, parse_json

ERR = re.compile(r'error|not found|не найден|invalid|failed', re.I)
VALUE_TYPES = {'ID', 'DATE', 'MD', 'TIME', 'NUM', 'VAL'}


def _nums(ents):
    out = []
    for t, n in ents:
        if t == 'NUM':
            v = _num(n)
            if v is not None:
                out.append(v)
    return out


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def derivable(x, pool):
    """Число получается одной операцией из недавних чисел или их суммой (агрегация — не галлюцинация)."""
    base = [p for p in pool if abs(p) < 1e5][-60:]
    small = [float(k) for k in range(2, 6)]
    for i, a in enumerate(base):
        for b in base[i:] + small:
            for v in (a + b, a - b, b - a, a * b, a / b if b else None):
                if v is not None and abs(v - x) < 0.011:
                    return True
    return len(base) > 1 and abs(sum(base) - x) < 0.011


def value_entities(v):
    """Значение аргумента JSON -> [(тип, нормализованное значение)]."""
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [('NUM', norm_num(str(v)))]
    if isinstance(v, (dict, list)):
        out = []
        for x in leaves(v):
            out += value_entities(x)
        return out
    s = str(v).strip()
    if s[:1] in '{[':
        inner = parse_json(s)
        if isinstance(inner, (dict, list)):
            return value_entities(inner)
    if s.isdigit():
        return [('ID', s)]
    ents = extract_entities(s)
    covered = sum(b - a for a, b, _, _ in ents)
    if ents and covered >= 0.6 * len(s):
        return [('VAL' if t == 'SYM' else t, n) for _, _, t, n in ents]
    if len(s) <= 40:
        return [('VAL', s.lower())] if s else []
    return [(t, n) for _, _, t, n in ents if t != 'SYM']


def leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)
    else:
        yield obj


def keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys(v)


def _sig(seg):
    try:
        return seg.name, json.dumps(seg.args, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return seg.name, str(seg.args)


def check(prompt: str, response: str) -> dict:
    ctx = parse(prompt)
    last_turn = max([s.turn for s in ctx.segs] or [0])
    resp = parse(response, turn0=last_turn)
    calls = [s for s in resp.segs if s.kind == 'call']

    # ---- множество сущностей контекста
    ctx_ents = {(t, n) for _, _, t, n in extract_entities(prompt)}
    for s in ctx.segs:
        if s.kind in ('call', 'result'):
            ctx_ents.update(value_entities(s.args))
    ctx_md = {n[5:] for t, n in ctx_ents if t == 'DATE'} | {n for t, n in ctx_ents if t == 'MD'}
    low = prompt.lower()

    # ---- недавние числа (последние два хода) — для проверки агрегации
    pool = []
    for s in ctx.segs:
        if s.kind not in ('policy', 'tools') and s.turn >= last_turn - 2:
            ents = value_entities(s.args) if s.kind in ('call', 'result') else \
                [(t, n) for _, _, t, n in extract_entities(s.text)]
            pool += _nums(ents)

    # ---- сущности ответа
    resp_ents = []
    for s in resp.segs:
        if s.kind == 'assistant':
            resp_ents += [(t, n) for _, _, t, n in extract_entities(s.text) if t != 'SYM']
        elif s.kind == 'call':
            resp_ents += value_entities(s.args)
    pool += _nums(resp_ents)

    unsupported = []
    for t, n in resp_ents:
        if t not in VALUE_TYPES:
            continue
        ok = (t, n) in ctx_ents
        if not ok and t in ('ID', 'NUM'):
            ok = ('ID', n) in ctx_ents or ('NUM', n) in ctx_ents
        if not ok and t == 'MD':
            ok = n in ctx_md
        if not ok and t == 'DATE':
            ok = n[5:] in ctx_md
        if not ok and t == 'NUM':
            x = _num(n)
            ok = x is not None and derivable(x, pool)
        if not ok and t == 'VAL':
            ok = len(n) > 3 and n in low
        if not ok:
            unsupported.append(f'{t}:{n}')

    # ---- инструменты, ключи, повторы
    bad_tools = [s.name for s in calls if s.name not in ctx.tools]
    bad_keys = sorted({k for s in calls if s.name in ctx.tools for k in keys(s.args) if k not in ctx.tools[s.name]})
    loop = False
    for s in calls:
        sig = _sig(s)
        for i, c in enumerate(ctx.segs):
            if c.kind == 'call' and _sig(c) == sig:
                nxt = next((r for r in ctx.segs[i + 1:] if r.kind == 'result' and r.name == c.name), None)
                if nxt is not None and ERR.search(nxt.text[:300]):
                    loop = True

    hot = bool(bad_tools) or loop or bool(unsupported)
    warm = bool(bad_keys)
    return dict(has_call=bool(calls), hot=hot, warm=warm, bad_tools=bad_tools, bad_keys=bad_keys,
                loop=loop, unsupported=sorted(set(unsupported)))
