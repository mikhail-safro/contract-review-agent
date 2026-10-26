"""Граф проверки договора на LangGraph.

    anonymize → split → guard → load_checklist → triage → verify ─┬→ legal_basis → human_review → report
                                                   ↑              │
                                                   └── повтор ────┘ (цитата не найдена в тексте)

Детерминированные шаги (обезличивание, разбивка, защита, проверка цитат)
написаны обычным кодом. LLM работает только там, где нужно понимание текста:
поиск рисков в пункте и выбор статьи закона.
"""

import json
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from contract_review_agent import llm as prompts
from contract_review_agent.anonymize import anonymize
from contract_review_agent.clauses import find_injections, sanitize, split_clauses
from contract_review_agent.knowledge import normalize_article_id
from contract_review_agent.report import build_report
from contract_review_agent.rules import check_quote, rule_candidates, rule_quote

INJECTION_CODE = "PROMPT_INJECTION"


class ReviewState(TypedDict, total=False):
    contract: str                 # исходный текст (остаётся локально)
    masked: str                   # обезличенный текст — только он уходит в LLM
    pii_map: dict[str, str]       # метка → исходное значение
    clauses: list[dict]           # [{id, text}]
    injections: list[dict]        # [{clause_id, sentence}]
    checklist: list[dict]
    pending: list[dict]           # пункты на (повторный) анализ: [{clause_id, feedback}]
    attempt: int
    findings: list[dict]
    rejected: list[dict]
    human: dict                   # решение человека
    stats: dict[str, Any]
    report: str


# ---------- MCP ----------

@asynccontextmanager
async def legal_kb_tools():
    """Запустить MCP-сервер legal-kb (stdio) и отдать его инструменты как LangChain tools.

    Одна сессия на весь прогон: сервер стартует один раз, а не на каждый вызов.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    client = MultiServerMCPClient({
        "legal-kb": {
            "command": sys.executable,
            "args": ["-m", "contract_review_agent.mcp_server"],
            "transport": "stdio",
        }
    })
    async with client.session("legal-kb") as session:
        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


def tool_json(result: Any) -> Any:
    """Ответ MCP-инструмента → Python-объекты (по одному JSON на блок)."""
    if isinstance(result, str):
        return json.loads(result)
    items = [json.loads(b["text"] if isinstance(b, dict) else b.text) for b in result]
    return items


# ---------- агент-юрист (подграф) ----------

def build_lawyer(llm: BaseChatModel, tools: list[BaseTool]):
    """Классический цикл агента: LLM решает, вызвать ли инструмент; результат возвращается ей же."""
    llm_with_tools = llm.bind_tools(tools)

    async def agent(state: MessagesState):
        msgs = [SystemMessage(prompts.LAWYER_SYSTEM)] + state["messages"]
        return {"messages": [await llm_with_tools.ainvoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


_ARTICLE_LINE = re.compile(r"СТАТЬЯ:\s*(.+)", re.IGNORECASE)
_WHY_LINE = re.compile(r"ПОЧЕМУ:\s*(.+)", re.IGNORECASE)


def parse_lawyer_answer(text: str) -> tuple[str | None, str]:
    m = _ARTICLE_LINE.search(text)
    article = normalize_article_id(m.group(1)) if m else normalize_article_id(text)
    why = _WHY_LINE.search(text)
    return article, (why.group(1).strip() if why else "")


def injection_finding(clause_id: str, quote: str, source: str) -> dict:
    explanation = ("В договор вставлена инструкция для ИИ. Предложение вырезано перед анализом; "
                   "документ стоит проверить вручную." if source == "guard" else
                   "LLM заметила в пункте обращение к ИИ или просьбу повлиять на проверку; "
                   "регулярные выражения его не нашли. Пункт стоит проверить вручную.")
    return {"id": f"{clause_id}:{INJECTION_CODE}", "code": INJECTION_CODE, "clause_id": clause_id,
            "title": "Попытка prompt injection в тексте договора", "severity": "high", "quote": quote,
            "grounding": "exact", "source": source, "status": "verified", "explanation": explanation}


# ---------- граф ----------

def build_graph(llm: BaseChatModel | None, tools: dict[str, BaseTool], *,
                checkpointer=None, hitl: bool = False, max_retries: int = 1):
    """llm=None — режим «только правила» (базовая линия для оценки качества)."""

    lawyer = build_lawyer(llm, [tools["search_law"], tools["get_law_article"]]) if llm else None
    law_cache: dict[str, dict] = {}  # код риска → статья; общий для всех прогонов этого графа

    def bump(state: ReviewState, **delta) -> dict:
        stats = dict(state.get("stats", {}))
        for k, v in delta.items():
            stats[k] = stats.get(k, 0) + v
        return stats

    async def anonymize_node(state: ReviewState):
        masked, pii_map = anonymize(state["contract"])
        return {"masked": masked, "pii_map": pii_map,
                "stats": {"started": time.time(), "pii_masked": len(pii_map), "llm_calls": 0}}

    async def split_node(state: ReviewState):
        return {"clauses": split_clauses(state["masked"])}

    async def guard_node(state: ReviewState):
        hits = find_injections(state["clauses"])
        findings = [injection_finding(h["clause_id"], h["sentence"], "guard") for h in hits]
        return {"injections": hits, "findings": findings, "rejected": []}

    async def load_checklist_node(state: ReviewState):
        checklist = tool_json(await tools["get_checklist"].ainvoke({}))
        pending = [{"clause_id": c["id"], "feedback": ""} for c in state["clauses"]]
        return {"checklist": checklist, "pending": pending, "attempt": 0}

    async def triage_node(state: ReviewState):
        checklist = state["checklist"]
        by_code = {c["code"]: c for c in checklist}
        clauses = {c["id"]: c for c in state["clauses"]}
        findings = list(state["findings"])
        known = {f["id"] for f in findings}
        calls = errors = 0

        if llm:
            schema = prompts.make_triage_schema([c["code"] for c in checklist])
            structured = llm.with_structured_output(schema, method="json_schema")
            system = prompts.TRIAGE_SYSTEM.format(checklist="\n".join(
                f'- {c["code"]}: {c["description"]}' for c in checklist))

        for task in state["pending"]:
            cid = task["clause_id"]
            text = sanitize(clauses[cid]["text"], [i for i in state["injections"] if i["clause_id"] == cid])
            hints = rule_candidates(text, checklist)
            risks: list[dict] = []
            if llm:
                hint = f"Подсказка правил (может быть неверной): {', '.join(hints) or 'нет'}."
                feedback = (f"\nВ прошлый раз цитата для {task['feedback']} не найдена в тексте пункта. "
                            "Цитируй дословно.") if task["feedback"] else ""
                try:
                    calls += 1
                    result = await structured.ainvoke([
                        SystemMessage(system),
                        HumanMessage(prompts.TRIAGE_USER.format(
                            clause_id=cid, clause_text=text, hint=hint, feedback=feedback)),
                    ])
                    if result.code != "NONE":
                        risks = [{"code": result.code, "quote": result.quote, "explanation": result.reasoning}]
                    # второй слой защиты: LLM замечает инструкции, которые не поймали регулярки
                    if result.ai_instruction and f"{cid}:{INJECTION_CODE}" not in known:
                        known.add(f"{cid}:{INJECTION_CODE}")
                        findings.append(injection_finding(cid, clauses[cid]["text"], "llm"))
                except Exception:  # невалидный JSON, таймаут — пункт пропускаем, но считаем
                    errors += 1
            else:
                risks = [{"code": code, "quote": rule_quote(text, by_code[code]),
                          "explanation": by_code[code]["description"]} for code in hints]

            for r in risks:
                fid = f'{cid}:{r["code"]}'
                if fid in known:
                    continue
                known.add(fid)
                item = by_code[r["code"]]
                findings.append({
                    "id": fid, "code": r["code"], "clause_id": cid, "title": item["title"],
                    "severity": item["severity"], "quote": r["quote"].strip(),
                    "explanation": r["explanation"].strip(), "source": "llm" if llm else "rules",
                    "rule_hint": r["code"] in hints, "status": "new",
                })
        return {"findings": findings, "stats": bump(state, llm_calls=calls, llm_errors=errors)}

    async def verify_node(state: ReviewState):
        """Проверка обычным кодом: цитата должна быть в тексте пункта."""
        clauses = {c["id"]: c["text"] for c in state["clauses"]}
        attempt = state["attempt"]
        findings, rejected, retry = [], list(state["rejected"]), {}
        for f in state["findings"]:
            if f["status"] != "new":
                findings.append(f)
                continue
            grounding = check_quote(f["quote"], clauses[f["clause_id"]])
            if grounding:
                findings.append({**f, "status": "verified", "grounding": grounding})
            elif attempt < max_retries:
                retry.setdefault(f["clause_id"], []).append(f["code"])
            else:
                rejected.append({**f, "status": "rejected", "reason": "цитата не найдена в тексте пункта"})
        pending = [{"clause_id": cid, "feedback": ", ".join(codes)} for cid, codes in retry.items()]
        return {"findings": findings, "rejected": rejected, "pending": pending, "attempt": attempt + 1,
                "stats": bump(state, retries=len(pending))}

    def route_after_verify(state: ReviewState) -> str:
        return "triage" if state["pending"] else "legal_basis"

    async def legal_basis_node(state: ReviewState):
        """Агент-юрист через MCP находит статью закона; ответ проверяется по базе."""
        by_code = {c["code"]: c for c in state["checklist"]}
        cache = law_cache
        findings, calls = [], 0
        for f in state["findings"]:
            if f["code"] == INJECTION_CODE:
                findings.append(f)
                continue
            if f["code"] not in cache:
                article, why, source = None, "", "checklist"
                if lawyer:
                    try:
                        out = await lawyer.ainvoke(
                            {"messages": [HumanMessage(prompts.LAWYER_USER.format(
                                title=by_code[f["code"]]["title"], quote=f["quote"]))]},
                            {"recursion_limit": 8})
                        calls += sum(isinstance(m, AIMessage) for m in out["messages"])
                        article, why = parse_lawyer_answer(out["messages"][-1].content)
                        source = "agent" if article else "checklist"
                        why = why if article else ""
                    except GraphRecursionError:
                        calls += 4
                article = article or by_code[f["code"]]["law_refs"][0]
                card = tool_json(await tools["get_law_article"].ainvoke({"article": article}))[0]
                cache[f["code"]] = {
                    "article": card["id"], "title": card["title"], "summary": card["summary"],
                    "why": why, "source": source,
                    "expected": card["id"] in by_code[f["code"]]["law_refs"],
                }
            findings.append({**f, "law": cache[f["code"]],
                             "recommendation": by_code[f["code"]]["recommendation"]})
        return {"findings": findings, "stats": bump(state, llm_calls=calls)}

    async def human_review_node(state: ReviewState):
        """Остановка для юриста: подтвердить или снять критичные замечания.

        interrupt() сохраняет состояние в checkpointer и завершает прогон;
        продолжение — graph.ainvoke(Command(resume={...}), config) с тем же thread_id.
        """
        critical = [f for f in state["findings"] if f["severity"] == "high"]
        if not hitl or not critical:
            return {"human": {"reviewed": False}}
        decision = interrupt({
            "question": "Подтвердите критичные замечания. Укажите id тех, которые нужно снять.",
            "findings": [{"id": f["id"], "title": f["title"], "clause_id": f["clause_id"],
                          "quote": f["quote"]} for f in critical],
        })
        dismissed = set(decision.get("dismissed", []))
        findings = [({**f, "status": "dismissed"} if f["id"] in dismissed else f) for f in state["findings"]]
        return {"findings": findings,
                "human": {"reviewed": True, "dismissed": sorted(dismissed), "comment": decision.get("comment", "")}}

    async def report_node(state: ReviewState):
        stats = dict(state["stats"])
        stats["seconds"] = round(time.time() - stats.pop("started"), 1)
        stats["model"] = prompts.model_name() if llm else "rules-only"
        return {"report": build_report(state, stats), "stats": stats}

    g = StateGraph(ReviewState)
    for name, fn in [("anonymize", anonymize_node), ("split", split_node), ("guard", guard_node),
                     ("load_checklist", load_checklist_node), ("triage", triage_node),
                     ("verify", verify_node), ("legal_basis", legal_basis_node),
                     ("human_review", human_review_node), ("report", report_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "anonymize")
    g.add_edge("anonymize", "split")
    g.add_edge("split", "guard")
    g.add_edge("guard", "load_checklist")
    g.add_edge("load_checklist", "triage")
    g.add_edge("triage", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"triage": "triage", "legal_basis": "legal_basis"})
    g.add_edge("legal_basis", "human_review")
    g.add_edge("human_review", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
