"""Извлечение сущностей регулярными выражениями: ID, даты, числа, email, телефоны."""
from __future__ import annotations

import re


# ---------------------------------------------------------------- сущности
MONTHS = {'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'ма': 5, 'июн': 6, 'июл': 7,
          'август': 8, 'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12}
ENT_RE = re.compile(
    r'(?P<email>[\w.+\-]+@[\w\-]+\.[\w.]+)'
    r'|(?P<date>\b\d{4}-\d{2}-\d{2}\b)(?:[ T]\d{2}:\d{2}(?::\d{2})?)?'
    r'|(?P<rudate>\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*)'
    r'|(?P<phone>\b\d{3}-\d{3}-\d{4}\b)'
    r'|(?P<time>\b\d{1,2}:\d{2}\b)'
    r'|(?P<snake>#?\b[A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)+\b)'
    r'|(?P<code>#?\b(?!\d+(?:st|nd|rd|th)\b)(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b)'
    r'|(?P<num>(?<![\w.])\$?\d{1,3}(?:[  ,]\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\$?\d+(?:\.\d+)?(?![\w]))'
)
LIST_MARK = re.compile(r'^(\s*)\d+[.)]\s', re.M)


def norm_num(s: str) -> str:
    s = s.replace('$', '').replace(' ', '').replace(' ', '').replace(',', '')
    try:
        v = float(s)
    except ValueError:
        return s
    return str(int(v)) if v == int(v) else f'{v:.2f}'.rstrip('0').rstrip('.')


def extract_entities(text: str):
    """-> список (start, end, type, norm). type: ID | SYM | DATE | MD | NUM | TIME."""
    text = LIST_MARK.sub(lambda m: ' ' * len(m.group(0)), text)   # "1. " в списках — не число (длина сохраняется)
    out = []
    for m in ENT_RE.finditer(text):
        k = m.lastgroup
        s = m.group(k)
        if k == 'email' or k == 'phone':
            out.append((m.start(), m.end(), 'ID', s.lower()))
        elif k == 'date':
            out.append((m.start(), m.end(), 'DATE', s))
        elif k == 'rudate':
            d, mon = s.split()[0], s.split()[1]
            mm = next(v for p, v in sorted(MONTHS.items(), key=lambda x: -len(x[0])) if mon.startswith(p))
            out.append((m.start(), m.end(), 'MD', f'{mm:02d}-{int(d):02d}'))
        elif k == 'time':
            out.append((m.start(), m.end(), 'TIME', s.zfill(5)))
        elif k == 'snake':
            t = 'ID' if re.search(r'\d', s) else 'SYM'
            out.append((m.start(), m.end(), t, s.lstrip('#').lower()))
        elif k == 'code':
            out.append((m.start(), m.end(), 'ID', s.lstrip('#').lower()))
        elif k == 'num':
            out.append((m.start(), m.end(), 'NUM', norm_num(s)))
    return out
