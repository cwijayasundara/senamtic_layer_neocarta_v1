"""Offline robustness tests for stream_chat_events: the UI must always receive a
final answer event (even on agent error), and the agent must run with a generous
recursion budget so heavy multi-subagent queries can complete."""

import json

from semantic_layer.config import settings
from semantic_layer.web import events as events_mod
from semantic_layer.web.events import stream_chat_events


# Minimal stand-ins. events.py dispatches on type(m).__name__, so the class
# names MUST be exactly these.
class AIMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class ToolMessage:
    def __init__(self, name, content, tool_call_id=None):
        self.name = name
        self.content = content
        self.tool_call_id = tool_call_id


class _FakeAgent:
    def __init__(self, steps, raise_exc=None, record=None):
        self._steps = steps
        self._raise = raise_exc
        self._record = record

    def stream(self, _inp, stream_mode=None, subgraphs=None, config=None):
        if self._record is not None:
            self._record["config"] = config
        for step in self._steps:
            yield step
        if self._raise:
            raise self._raise


def test_stream_emits_answer_even_when_agent_raises(monkeypatch):
    ai = AIMessage(tool_calls=[{"id": "t1", "name": "run_sql",
                                "args": {"source": "sales_pg", "sql": "SELECT 1"}}])
    tool = ToolMessage("run_sql", json.dumps({"columns": ["n"], "rows": [[5]]}), "t1")
    steps = [(("orch",), {"agent": {"messages": [ai]}}),
             (("orch",), {"tools": {"messages": [tool]}})]
    fake = _FakeAgent(steps, raise_exc=RuntimeError("recursion blew up"))
    monkeypatch.setattr(events_mod, "build_agent", lambda: fake)

    out = list(stream_chat_events("q"))
    assert out[-1]["type"] == "answer"                 # never leave the UI hanging
    assert "recursion blew up" in out[-1]["content"] or "stopped early" in out[-1]["content"]
    assert out[-1]["sql_runs"][0]["sql"] == "SELECT 1"  # partial provenance survives


def test_stream_passes_recursion_limit(monkeypatch):
    rec: dict = {}
    fake = _FakeAgent([], record=rec)
    monkeypatch.setattr(events_mod, "build_agent", lambda: fake)

    list(stream_chat_events("q"))
    assert rec["config"]["recursion_limit"] == settings.agent_recursion_limit
    assert settings.agent_recursion_limit >= 50
