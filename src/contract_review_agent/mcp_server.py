"""MCP-сервер «legal-kb»: чек-лист рисков и база статей закона.

Запуск: `uv run cra mcp` (транспорт stdio). Этот же сервер можно подключить
к Claude Desktop или к любому другому MCP-клиенту.
"""

from mcp.server.fastmcp import FastMCP

from contract_review_agent import knowledge

mcp = FastMCP("legal-kb", log_level="WARNING")


@mcp.tool()
def get_checklist() -> list[dict]:
    """Чек-лист рисков договора с позиции Заказчика: код, название, критичность, признаки, статьи закона."""
    return knowledge.checklist()


@mcp.tool()
def search_law(query: str, top_k: int = 3) -> list[dict]:
    """Найти статьи ГК РФ и АПК РФ по описанию ситуации, например «неустойка без ограничения».

    Возвращает до top_k статей: номер (id), название и краткий пересказ.
    """
    return knowledge.search_law(query, top_k)


@mcp.tool()
def get_law_article(article: str) -> dict:
    """Получить статью по номеру, например «ГК 333» или «АПК 37». Если статьи нет в базе — found=false."""
    art = knowledge.get_article(article)
    return {"found": True, **art} if art else {"found": False, "article": article}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
