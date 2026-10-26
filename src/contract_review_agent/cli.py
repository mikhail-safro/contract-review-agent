"""Командная строка.

    cra review договор.txt [--hitl] [--rules-only]   проверить договор
    cra resume --thread <id>                          продолжить проверку после паузы
    cra eval [--rules-only]                           оценка качества на размеченных договорах
    cra graph                                         схема графа (Mermaid)
    cra mcp                                           запустить MCP-сервер legal-kb (stdio)
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from contract_review_agent.graph import build_graph, legal_kb_tools
from contract_review_agent.llm import make_llm

DB = ".checkpoints.sqlite"


async def _run(graph, payload, config) -> dict:
    """Прогнать граф; при остановке на human_review спросить человека."""
    while True:
        result = await graph.ainvoke(payload, config)
        if "__interrupt__" not in result:
            return result
        request = result["__interrupt__"][0].value
        print(f"\n⏸  {request['question']}")
        for n, f in enumerate(request["findings"], 1):
            print(f"  {n}. п. {f['clause_id']} — {f['title']}\n     «{f['quote']}»")
        answer = input("\nНомера замечаний, которые снять (через запятую; Enter — подтвердить все; "
                       "«потом» — отложить): ").strip()
        if answer.lower() in {"потом", "later"}:
            thread = config["configurable"]["thread_id"]
            print(f"\nСостояние сохранено. Продолжить: cra resume --thread {thread}")
            return {}
        picked = {int(x) for x in answer.replace(" ", "").split(",") if x.isdigit()}
        dismissed = [f["id"] for n, f in enumerate(request["findings"], 1) if n in picked]
        comment = input("Комментарий юриста (необязательно): ").strip() if dismissed else ""
        payload = Command(resume={"dismissed": dismissed, "comment": comment})


def _save(result: dict, out: Path) -> None:
    if not result:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["report"], encoding="utf-8")
    found = sum(f["status"] == "verified" for f in result["findings"])
    print(f"\n✅ Отчёт: {out}  (замечаний: {found}, время: {result['stats']['seconds']} с)")


async def review(args) -> None:
    path = Path(args.file)
    thread = args.thread or f"{path.stem}-{int(time.time())}"
    out = Path(args.out or f"reports/{path.stem}.md")
    async with legal_kb_tools() as tools, AsyncSqliteSaver.from_conn_string(args.db) as saver:
        llm = None if args.rules_only else make_llm()
        graph = build_graph(llm, tools, checkpointer=saver, hitl=args.hitl)
        config = {"configurable": {"thread_id": thread}}
        print(f"Проверяю {path.name} (thread: {thread})…")
        result = await _run(graph, {"contract": path.read_text(encoding="utf-8")}, config)
        _save(result, out)


async def resume(args) -> None:
    async with legal_kb_tools() as tools, AsyncSqliteSaver.from_conn_string(args.db) as saver:
        graph = build_graph(make_llm(), tools, checkpointer=saver, hitl=True)
        config = {"configurable": {"thread_id": args.thread}}
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            sys.exit(f"Для thread {args.thread} нет незавершённой проверки.")
        # Повторно показать вопрос: прогон с None продолжает граф с места остановки
        result = await _run(graph, None, config)
        _save(result, Path(args.out or f"reports/{args.thread}.md"))


async def show_graph(_args) -> None:
    from contract_review_agent.graph import build_lawyer

    async with legal_kb_tools() as tools:
        print("```mermaid")
        print(build_graph(None, tools).get_graph().draw_mermaid())
        print("```\n\nАгент-юрист (подграф узла legal_basis):\n\n```mermaid")
        fake = _ToolFake(responses=["-"])
        print(build_lawyer(fake, [tools["search_law"], tools["get_law_article"]]).get_graph().draw_mermaid())
        print("```")


class _ToolFake(FakeListChatModel):
    """Модель-заглушка: нужна только чтобы отрисовать подграф без запуска LLM."""

    def bind_tools(self, tools, **kwargs):
        return self


def main() -> None:
    parser = argparse.ArgumentParser(prog="cra", description="Агент проверки договоров (LangGraph + MCP)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("review", help="проверить договор")
    p.add_argument("file")
    p.add_argument("--hitl", action="store_true", help="остановиться для подтверждения критичных замечаний")
    p.add_argument("--rules-only", action="store_true", help="без LLM, только правила (базовая линия)")
    p.add_argument("--thread", help="id прогона для сохранения состояния")
    p.add_argument("--out", help="куда сохранить отчёт (по умолчанию reports/<имя>.md)")
    p.add_argument("--db", default=DB)

    p = sub.add_parser("resume", help="продолжить проверку, остановленную на подтверждении")
    p.add_argument("--thread", required=True)
    p.add_argument("--out")
    p.add_argument("--db", default=DB)

    p = sub.add_parser("eval", help="оценка качества на размеченных договорах")
    p.add_argument("--rules-only", action="store_true")
    p.add_argument("--only", help="подстрока имени договора")
    p.add_argument("--set", default="dev", choices=["dev", "holdout"],
                   help="dev — договоры, на которых настраивались правила; holdout — отложенная выборка")

    sub.add_parser("graph", help="схема графа в формате Mermaid")
    sub.add_parser("mcp", help="запустить MCP-сервер legal-kb (stdio)")

    args = parser.parse_args()
    if args.cmd == "mcp":
        from contract_review_agent.mcp_server import main as mcp_main
        return mcp_main()
    if args.cmd == "eval":
        from contract_review_agent.evaluate import main as eval_main
        return asyncio.run(eval_main(args))
    handler = {"review": review, "resume": resume, "graph": show_graph}[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
