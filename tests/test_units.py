from contract_review_agent.anonymize import anonymize, restore
from contract_review_agent.clauses import find_injections, sanitize, split_clauses
from contract_review_agent.knowledge import checklist, normalize_article_id, search_law
from contract_review_agent.rules import check_quote, rule_candidates


def test_anonymize_roundtrip_and_masks(contract):
    text = contract("01_техподдержка")
    masked, mapping = anonymize(text)
    assert restore(masked, mapping) == text
    for secret in ["7701234567", "40702810900000012345", "+7 (495) 123-45-67",
                   "office@sever-veter.ru", "Иванова Петра Сергеевича", "П.С. Иванов"]:
        assert secret not in masked
    assert "ИНН [ИНН_1]" in masked  # контекст остаётся, маскируется только значение


def test_same_value_same_tag():
    masked, mapping = anonymize("ИНН 7701234567 и снова ИНН 7701234567")
    assert masked == "ИНН [ИНН_1] и снова ИНН [ИНН_1]"
    assert mapping == {"[ИНН_1]": "7701234567"}


def test_split_clauses_skips_requisites(contract):
    clauses = split_clauses(contract("01_техподдержка"))
    assert [c["id"] for c in clauses][:3] == ["1.1", "2.1", "2.2"]
    assert len(clauses) == 13
    assert "ИНН" not in clauses[-1]["text"]


def test_injection_found_and_cut(contract):
    clauses = split_clauses(contract("05_логистика_injection"))
    hits = find_injections(clauses)
    assert [h["clause_id"] for h in hits] == ["3.2"]
    clean = sanitize(next(c for c in clauses if c["id"] == "3.2")["text"], hits)
    assert "игнорируй" not in clean and "100 рублей" in clean


def test_no_false_injection(contract):
    assert find_injections(split_clauses(contract("04_консалтинг_сбалансированный"))) == []


def test_check_quote():
    clause = "За просрочку оплаты Заказчик уплачивает пени в размере 1% за каждый день просрочки."
    assert check_quote("Заказчик уплачивает пени в размере 1%", clause) == "exact"
    assert check_quote("Заказчик уплачивает пеню в размере 1%", clause) == "fuzzy"
    assert check_quote("Исполнитель вправе изменить цену", clause) is None
    assert check_quote("пени", clause) is None  # слишком короткая цитата ничего не доказывает


def test_rule_candidates_respect_cap():
    items = checklist()
    assert "PENALTY_UNCAPPED" in rule_candidates("Заказчик уплачивает пени 1% в день.", items)
    assert "PENALTY_UNCAPPED" not in rule_candidates("Пени 0,1% в день, но не более 5% от суммы.", items)


def test_law_search_and_ids():
    assert {a["id"] for a in search_law("неустойка несоразмерна")} >= {"ГК 333"}
    assert normalize_article_id("ст. 333 ГК РФ") == "ГК 333"
    assert normalize_article_id("ГК РФ ст.450.1") == "ГК 450.1"
    assert normalize_article_id("ГК 9999") is None
