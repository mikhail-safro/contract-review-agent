"""Граф целиком: настоящий MCP-сервер (stdio) + модель-заглушка."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from contract_review_agent.graph import INJECTION_CODE, build_graph, legal_kb_tools, tool_json
from tests.conftest import FakeLLM


async def test_mcp_tools():
    async with legal_kb_tools() as tools:
        assert set(tools) == {"get_checklist", "search_law", "get_law_article"}
        assert len(tool_json(await tools["get_checklist"].ainvoke({}))) == 9
        assert tool_json(await tools["get_law_article"].ainvoke({"article": "ст. 782 ГК"}))[0]["id"] == "ГК 782"
        assert tool_json(await tools["get_law_article"].ainvoke({"article": "ГК 1"}))[0]["found"] is False


async def test_rules_only(contract):
    async with legal_kb_tools() as tools:
        result = await build_graph(None, tools).ainvoke({"contract": contract("05_логистика_injection")})
    codes = {(f["clause_id"], f["code"]) for f in result["findings"]}
    assert ("3.1", "PENALTY_UNCAPPED") in codes
    assert ("3.2", INJECTION_CODE) in codes
    assert "7701" not in result["masked"] and "pavlov@" not in result["masked"]
    assert "# Отчёт о проверке договора" in result["report"]


async def test_llm_path_uses_lawyer_agent(contract):
    async with legal_kb_tools() as tools:
        result = await build_graph(FakeLLM(), tools).ainvoke({"contract": contract("01_техподдержка")})
    verified = [f for f in result["findings"] if f["status"] == "verified"]
    assert verified and all(f["law"]["source"] == "agent" for f in verified)
    assert result["stats"]["llm_calls"] >= 13  # по вызову на пункт + агент-юрист


async def test_made_up_quote_is_retried(contract):
    async with legal_kb_tools() as tools:
        result = await build_graph(FakeLLM(bad_quotes=True), tools, max_retries=1).ainvoke(
            {"contract": contract("02_клининг")})
    assert result["stats"]["retries"] >= 1
    assert all(f["grounding"] == "exact" for f in result["findings"] if f["status"] == "verified")


async def test_made_up_quote_is_rejected_without_retries(contract):
    async with legal_kb_tools() as tools:
        result = await build_graph(FakeLLM(bad_quotes=True), tools, max_retries=0).ainvoke(
            {"contract": contract("02_клининг")})
    assert result["rejected"] and not [f for f in result["findings"] if f["status"] == "verified"]


async def test_human_in_the_loop(contract):
    config = {"configurable": {"thread_id": "t1"}}
    async with legal_kb_tools() as tools:
        graph = build_graph(FakeLLM(), tools, checkpointer=InMemorySaver(), hitl=True)
        paused = await graph.ainvoke({"contract": contract("01_техподдержка")}, config)
        assert "__interrupt__" in paused
        question = paused["__interrupt__"][0].value
        drop = question["findings"][0]["id"]
        # состояние сохранено в checkpointer: продолжаем тот же thread
        assert (await graph.aget_state(config)).next == ("human_review",)
        done = await graph.ainvoke(Command(resume={"dismissed": [drop], "comment": "согласовано"}), config)
    assert next(f for f in done["findings"] if f["id"] == drop)["status"] == "dismissed"
    assert "## Снято юристом" in done["report"] and "согласовано" in done["report"]
