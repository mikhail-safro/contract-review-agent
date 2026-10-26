"""Разбивка договора на пункты и защита от prompt injection."""

import re

_CLAUSE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)\.?\s+(.*)$")
_SECTION = re.compile(r"^\s*\d+\.\s+[А-ЯЁ\s,«»-]{4,}$")

# Признаки инструкций, адресованных модели. Документ — это данные,
# поэтому любая попытка «поговорить с ИИ» из текста договора подозрительна.
_INJECTION = [
    r"игнорир\w*\s+(все\s+)?(предыдущ|прежн|ранее|данн)",
    r"ignore\s+(all\s+)?(previous|prior|above)",
    r"(для|к)\s+(ии|ai|нейросет\w*|модел\w*|ассистент\w*|языков\w* модел\w*)\b",
    r"system\s*prompt|системн\w+ (промпт|инструкц)",
    r"(сообщи|ответь|укажи|напиши)\w*,?\s+что\s+(рисков|замечаний|нарушений)\s+нет",
    r"не\s+(сообщай|указывай|упоминай)\w*",
    r"ты\s+(теперь|—|-)\s",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def split_clauses(text: str) -> list[dict]:
    """Пункты вида «5.2. Текст…»; продолжение строки приклеивается к пункту.

    Заголовки разделов и текст до первого пункта (преамбула) не являются
    условиями договора и в проверку не идут.
    """
    clauses: list[dict] = []
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _CLAUSE.match(line)
        if m:
            current = {"id": m.group(1), "text": m.group(2).strip()}
            clauses.append(current)
        elif _SECTION.match(line):
            current = None  # новый раздел: реквизиты и подписи к пунктам не приклеиваем
        elif current:
            current["text"] += " " + line.strip()
    return clauses


def find_injections(clauses: list[dict]) -> list[dict]:
    """Найти предложения-инструкции, вставленные в договор."""
    hits = []
    for c in clauses:
        for sent in _SENTENCE.findall(c["text"]):
            if _INJECTION_RE.search(sent):
                hits.append({"clause_id": c["id"], "sentence": sent.strip()})
    return hits


def sanitize(clause_text: str, injections: list[dict]) -> str:
    """Вырезать подозрительные предложения из текста, который увидит LLM."""
    for hit in injections:
        clause_text = clause_text.replace(hit["sentence"], "[удалено: подозрение на prompt injection]")
    return clause_text
