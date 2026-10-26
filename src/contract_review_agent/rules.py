"""Детерминированная часть: кандидаты рисков по правилам и проверка цитат.

Правила дают кандидатов с высокой полнотой и низкой точностью; LLM их
подтверждает или отклоняет и может добавить риск, который правила не нашли.
Режим «только правила» используется как базовая линия в оценке качества.
"""

import re
from difflib import SequenceMatcher

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "ё": "е", "Ё": "Е"})


def rule_candidates(clause_text: str, checklist: list[dict]) -> list[str]:
    """Коды рисков, признаки которых есть в тексте пункта."""
    low = clause_text.lower()
    codes = []
    for item in checklist:
        if not any(re.search(s, low) for s in item["signals"]):
            continue
        if any(neg in low for neg in item.get("negative_signals", [])):
            continue
        codes.append(item["code"])
    return codes


def rule_quote(clause_text: str, checklist_item: dict) -> str:
    """Для режима «только правила»: предложение, в котором сработал признак."""
    for sent in re.findall(r"[^.;]+[.;]?", clause_text):
        if any(re.search(s, sent.lower()) for s in checklist_item["signals"]):
            return sent.strip()
    return clause_text[:200]


def _norm(text: str) -> str:
    return _WS.sub(" ", text.translate(_QUOTES)).strip(" .;,\"").lower()


def check_quote(quote: str, clause_text: str) -> str | None:
    """Проверить, что цитата действительно взята из пункта.

    Возвращает «exact», «fuzzy» (совпадение ≥ 90% символов подряд идущими
    блоками — LLM поменяла падеж или пунктуацию) или None — цитата выдумана.
    """
    q, c = _norm(quote), _norm(clause_text)
    if len(q) < 10:
        return None
    if q in c:
        return "exact"
    matcher = SequenceMatcher(None, q, c, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= 4)
    return "fuzzy" if matched / len(q) >= 0.9 else None
