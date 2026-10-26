"""Обезличивание персональных и банковских данных перед отправкой текста в LLM.

Правила — регулярные выражения: они быстрые, предсказуемые и проверяются тестами.
Каждое найденное значение заменяется меткой вида [ФИО_1]; таблица замен
хранится в состоянии графа, и в итоговом отчёте значения восстанавливаются.
"""

import re

# Порядок важен: сначала длинные числовые реквизиты, затем короткие,
# чтобы 20-значный счёт не распознался как ИНН. Маскируется группа «v»,
# контекст вокруг неё («ИНН », «р/с ») остаётся в тексте.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"(?P<v>[\w.+-]+@[\w-]+\.[\w.-]*\w)")),
    ("ТЕЛЕФОН", re.compile(r"(?P<v>(?:\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2})\b")),
    ("СЧЁТ", re.compile(r"\b(?P<v>\d{20})\b")),
    ("ОГРН", re.compile(r"ОГРН(?:ИП)?\s(?P<v>\d{13}(?:\d{2})?)\b")),
    ("ИНН", re.compile(r"ИНН\s(?P<v>\d{10}(?:\d{2})?)\b")),
    ("КПП", re.compile(r"КПП\s(?P<v>\d{9})\b")),
    ("БИК", re.compile(r"БИК\s(?P<v>\d{9})\b")),
    ("ПАСПОРТ", re.compile(r"паспорт\w*[^\d]{0,20}(?P<v>\d{2}\s?\d{2}\s?№?\s?\d{6})\b", re.IGNORECASE)),
    # ФИО полностью: Фамилия Имя Отчество (отчество — по окончанию, в любом падеже)
    ("ФИО", re.compile(r"\b(?P<v>[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+?(?:вич|вн|ичн|ич)[а-яё]{0,3})\b")),
    # ФИО с инициалами: Иванов И.И. или И.И. Иванов
    ("ФИО", re.compile(r"(?P<v>\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+)")),
]


def anonymize(text: str) -> tuple[str, dict[str, str]]:
    """Вернуть текст с метками и таблицу «метка → исходное значение».

    Одинаковые значения получают одинаковую метку, поэтому LLM видит,
    что в разных пунктах речь об одном и том же лице.
    """
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    counters: dict[str, int] = {}

    for kind, pattern in _PATTERNS:
        def repl(m: re.Match, kind: str = kind) -> str:
            value = m.group("v")
            if value not in reverse:
                counters[kind] = counters.get(kind, 0) + 1
                tag = f"[{kind}_{counters[kind]}]"
                reverse[value] = tag
                mapping[tag] = value
            whole, offset = m.group(0), m.start("v") - m.start(0)
            return whole[:offset] + reverse[value] + whole[offset + len(value):]

        text = pattern.sub(repl, text)
    return text, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    for tag, value in mapping.items():
        text = text.replace(tag, value)
    return text
