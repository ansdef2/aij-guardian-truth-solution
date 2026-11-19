"""Разбор транскрипта prompt/response на сегменты.

Поддерживаются оба формата разметки:
  * формат данных:  ⟦ASSISTANT · ход N⟧ ... \t→ TOOL_CALL name: {...} / \t← TOOL_RESPONSE name: ...
  * формат из ТЗ:   ⟦ASSISTANT_TOOL_CALL name="X"⟧ {...} / ⟦TOOL_RESULT name="X" requestor="..."⟧ ...
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

BLOCK = re.compile(
    r'^⟦(SYSTEM|USER|ASSISTANT_TOOL_CALL|USER_TOOL_CALL|TOOL_RESULT|ASSISTANT)([^⟧]*)⟧[ \t]*$', re.M)
CALL = re.compile(r'^\s*→ TOOL_CALL ([\w\-.]+):\s?(.*)$')
RESULT = re.compile(r'^\s*← TOOL_RESPONSE ([\w\-.]+)(?:\s*\[[A-Z_]+\])?:\s?(.*)$')   # [ERROR] — ошибка среды
NAME_ATTR = re.compile(r'name="([^"]+)"')
TOOL_LINE = re.compile(r'^- ([\w\-.]+) — ')
PARAM_LINE = re.compile(r'^\s{2,}(?:· )?([\w\-.]+): ([\w\[\]]+)(!?)')


@dataclass
class Seg:
    kind: str            # policy | tools | user | assistant | call | result
    text: str
    turn: int
    name: str | None = None
    args: object = None   # разобранный JSON (для call/result)


@dataclass
class Transcript:
    segs: list[Seg] = field(default_factory=list)
    tools: dict[str, set[str]] = field(default_factory=dict)   # инструмент -> допустимые ключи


def parse_json(s: str):
    s = s.strip()
    if not s:
        return None
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:
            pass
    return s


def parse_tools(text: str) -> dict[str, set[str]]:
    tools: dict[str, set[str]] = {}
    cur = None
    for line in text.splitlines():
        m = TOOL_LINE.match(line)
        if m:
            cur = m.group(1)
            tools[cur] = set()
            continue
        m = PARAM_LINE.match(line)
        if m and cur:
            tools[cur].add(m.group(1))
    return tools


def _assistant_body(body: str, turn: int, out: list[Seg]):
    """Тело хода ассистента: текст + вызовы инструментов + их результаты."""
    text_lines, cur, items = [], None, []

    def flush():
        nonlocal cur
        if cur is not None:
            cur.args = parse_json(cur.text)
            items.append(cur)
            cur = None

    for line in body.splitlines():
        mc, mr = CALL.match(line), RESULT.match(line)
        if mc or mr:
            flush()
            m = mc or mr
            cur = Seg('call' if mc else 'result', m.group(m.lastindex), turn, name=m.group(1))
            if mr and '[ERROR]' in line.split(':', 1)[0]:
                cur.text = 'Error: ' + cur.text
        elif cur is not None:
            cur.text += '\n' + line
        else:
            text_lines.append(line)
    flush()
    txt = '\n'.join(text_lines).strip()
    if txt:
        out.append(Seg('assistant', txt, turn))
    out.extend(items)


def parse(text: str, turn0: int = 0) -> Transcript:
    tr = Transcript()
    marks = list(BLOCK.finditer(text))
    turn = turn0
    if not marks and text.strip():                       # ответ без маркеров
        _assistant_body(text, turn, tr.segs)
        return tr
    for i, m in enumerate(marks):
        role, attrs = m.group(1), m.group(2)
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip('\n')
        if role == 'SYSTEM':
            k = body.find('[AVAILABLE TOOLS]')
            policy, tools = (body, '') if k < 0 else (body[:k], body[k:])
            tr.segs.append(Seg('policy', policy, turn))
            if tools:
                tr.segs.append(Seg('tools', tools, turn))
                tr.tools.update(parse_tools(tools))
        elif role == 'USER':
            turn += 1
            tr.segs.append(Seg('user', body.strip(), turn))
        elif role == 'ASSISTANT':
            turn += 1
            _assistant_body(body, turn, tr.segs)
        elif role in ('ASSISTANT_TOOL_CALL', 'USER_TOOL_CALL'):
            nm = NAME_ATTR.search(attrs)
            tr.segs.append(Seg('call', body, turn, name=nm.group(1) if nm else None, args=parse_json(body)))
        elif role == 'TOOL_RESULT':
            nm = NAME_ATTR.search(attrs)
            tr.segs.append(Seg('result', body, turn, name=nm.group(1) if nm else None, args=parse_json(body)))
    return tr
