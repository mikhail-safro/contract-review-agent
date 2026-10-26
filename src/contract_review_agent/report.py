"""Итоговый отчёт в Markdown. Персональные данные в нём восстанавливаются."""

from contract_review_agent.anonymize import restore

SEVERITY = {"high": "🔴 высокая", "medium": "🟡 средняя", "low": "🟢 низкая"}
_ORDER = {"high": 0, "medium": 1, "low": 2}


def _clause_key(cid: str) -> tuple:
    return tuple(int(p) for p in cid.split("."))


def build_report(state: dict, stats: dict) -> str:
    pii = state.get("pii_map", {})
    active = [f for f in state["findings"] if f["status"] == "verified" and f["code"] != "PROMPT_INJECTION"]
    active.sort(key=lambda f: (_ORDER[f["severity"]], _clause_key(f["clause_id"])))
    injections = [f for f in state["findings"] if f["code"] == "PROMPT_INJECTION"]
    dismissed = [f for f in state["findings"] if f["status"] == "dismissed"]
    rejected = state.get("rejected", [])

    out = ["# Отчёт о проверке договора", ""]
    out.append(f"Позиция: **Заказчик**. Модель: `{stats['model']}`. Пунктов проверено: {len(state['clauses'])}. "
               f"Время: {stats['seconds']} с. Вызовов LLM: {stats.get('llm_calls', 0)}.")
    out.append("")
    if injections:
        out.append(f"> ⚠️ В документе найдена попытка prompt injection ({len(injections)}). Подробности — в разделе «Безопасность».")
        out.append("")

    out += ["## Сводка", ""]
    if active:
        out += ["| № | Критичность | Риск | Пункт | Статья |", "|---|---|---|---|---|"]
        for n, f in enumerate(active, 1):
            out.append(f"| {n} | {SEVERITY[f['severity']]} | {f['title']} | {f['clause_id']} | {f['law']['article']} |")
    else:
        out.append("Рисков по чек-листу не найдено.")
    out.append("")

    if active:
        out += ["## Замечания", ""]
        for n, f in enumerate(active, 1):
            law = f["law"]
            out.append(f"### {n}. {f['title']} — п. {f['clause_id']}")
            out.append("")
            out.append(f"- **Цитата:** «{restore(f['quote'], pii)}»" + (" *(≈, цитата неточная)*" if f["grounding"] == "fuzzy" else ""))
            out.append(f"- **Почему это риск:** {restore(f['explanation'], pii)}")
            out.append(f"- **Закон:** {law['article']} «{law['title']}». {law['summary']}"
                       + (f" *Комментарий агента: {law['why']}*" if law.get("why") else ""))
            out.append(f"- **Что предложить:** {f['recommendation']}")
            src = {"llm": "LLM", "rules": "правило"}[f["source"]]
            hint = ", правило тоже сработало" if f.get("rule_hint") and f["source"] == "llm" else ""
            out.append(f"- *Нашёл: {src}{hint}; статью подобрал: {'агент через MCP' if law['source'] == 'agent' else 'чек-лист'}.*")
            out.append("")

    if injections:
        out += ["## Безопасность", ""]
        for f in injections:
            how = ("найдено правилами, предложение вырезано из текста для LLM" if f["source"] == "guard"
                   else "замечено LLM (правила не сработали)")
            out.append(f"- п. {f['clause_id']}: «{restore(f['quote'], pii)}» — {how}.")
        out.append("")

    if dismissed:
        human = state.get("human", {})
        out += ["## Снято юристом", ""]
        out += [f"- {f['title']} — п. {f['clause_id']}" for f in dismissed]
        if human.get("comment"):
            out.append(f"- Комментарий: {human['comment']}")
        out.append("")

    if rejected:
        out += ["## Отброшено автоматической проверкой", ""]
        out += [f"- {f['title']} — п. {f['clause_id']}: {f['reason']}" for f in rejected]
        out.append("")

    out += ["---", "",
            f"Перед отправкой в LLM обезличено значений: {stats.get('pii_masked', 0)} (ФИО, ИНН, счета, телефоны, email). "
            "Отчёт — помощь юристу, а не юридическое заключение."]
    return "\n".join(out) + "\n"
