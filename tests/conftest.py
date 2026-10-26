"""Модель-заглушка: тесты графа идут без Ollama, быстро и детерминированно."""

import json
import re
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM(BaseChatModel):
    """Ведёт себя как LLM в двух ролях графа.

    triage: берёт первую подсказку правил как код риска, цитирует пункт;
            при bad_quotes=True сначала выдумывает цитату (проверка повтора).
    юрист:  сначала вызывает search_law, потом отвечает первой найденной статьёй.
    """

    bad_quotes: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def answer(messages):
            text = messages[-1].content
            clause = re.search(r"<clause>\n(.*)\n</clause>", text, re.S).group(1)
            hint = re.search(r"Подсказка правил[^:]*: ([A-Z_]+)", text)
            retry = "Цитируй дословно" in text
            quote = "этой фразы в пункте нет совсем" if self.bad_quotes and not retry else clause
            return schema(reasoning="тест", code=hint.group(1) if hint else "NONE",
                          quote=quote, ai_instruction=False)
        return RunnableLambda(answer)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            content = last.content
            first = content[0]["text"] if isinstance(content, list) else content
            found = json.loads(first)
            found = found[0] if isinstance(found, list) else found
            msg = AIMessage(content=f"СТАТЬЯ: {found['id']}\nПОЧЕМУ: тест")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "search_law", "args": {"query": "неустойка"},
                                                     "id": "call_1", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def contract():
    return lambda name: (ROOT / "evals" / "contracts" / f"{name}.txt").read_text(encoding="utf-8")
