"""База знаний: чек-лист рисков и краткие пересказы статей ГК РФ и АПК РФ.

Модуль используется MCP-сервером. Агент не читает файлы напрямую,
а обращается к знаниям через инструменты MCP.
"""

import json
import re
from functools import lru_cache
from importlib import resources

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _load(name: str) -> list[dict]:
    text = resources.files("contract_review_agent.data").joinpath(name).read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache
def checklist() -> list[dict]:
    return _load("checklist.json")


@lru_cache
def law_kb() -> dict[str, dict]:
    return {a["id"]: a for a in _load("law_kb.json")}


def normalize_article_id(raw: str) -> str | None:
    """«ст. 333 ГК РФ», «ГК РФ ст.333», «гк 333» → «ГК 333»."""
    m = re.search(r"(ГК|АПК)", raw, re.IGNORECASE)
    n = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m or not n:
        return None
    article_id = f"{m.group(1).upper()} {n.group(1)}"
    return article_id if article_id in law_kb() else None


def _stems(text: str) -> set[str]:
    # Грубый стемминг: первые 5 букв слова. Для русского языка этого хватает,
    # чтобы «неустойку» и «неустойка» совпали, и не нужна морфология.
    return {w.lower()[:5] for w in _WORD.findall(text) if len(w) >= 4}


def search_law(query: str, top_k: int = 3) -> list[dict]:
    q = _stems(query)
    scored = []
    for art in law_kb().values():
        score = (
            2 * len(q & _stems(" ".join(art["keywords"])))
            + 2 * len(q & _stems(art["title"]))
            + len(q & _stems(art["summary"]))
        )
        if score:
            scored.append((score, art["id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [_public(law_kb()[i]) for _, i in scored[:top_k]]


def get_article(article: str) -> dict | None:
    article_id = normalize_article_id(article)
    return _public(law_kb()[article_id]) if article_id else None


def _public(art: dict) -> dict:
    return {"id": art["id"], "title": art["title"], "summary": art["summary"]}
