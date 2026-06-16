"""Translate the agent's streamed steps into UI events + a highlight node-set."""

import json
from typing import Iterator

from semantic_layer.agent.build import build_agent


def _collect_highlight(tool_name: str, content: str, highlight: set[str]) -> None:
    """Accumulate graph node ids to light up, from specific tool results."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return
    if tool_name == "get_join_path" and isinstance(data, dict):
        for tid in data.get("tables", []):
            highlight.add(tid)
    elif tool_name == "search_catalog" and isinstance(data, list):
        for hit in data:
            tid = hit.get("table_id", "")
            if tid.startswith("table:"):
                highlight.add(tid)
    elif tool_name == "search_documents" and isinstance(data, list):
        for hit in data:
            if hit.get("doc_id"):
                highlight.add(hit["doc_id"])


def stream_chat_events(question: str) -> Iterator[dict]:
    """Yield UI events: {type: tool_call|tool_result|answer, ...}."""
    agent = build_agent()
    highlight: set[str] = set()
    final = ""
    for ns, chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates", subgraphs=True,
    ):
        scope = ns[-1].split(":")[0] if ns else "root"
        for _node, update in chunk.items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for m in messages:
                for call in getattr(m, "tool_calls", None) or []:
                    yield {"type": "tool_call", "scope": scope,
                           "name": call.get("name"), "args": call.get("args", {})}
                if type(m).__name__ == "ToolMessage":
                    name = getattr(m, "name", "")
                    content = str(getattr(m, "content", ""))
                    _collect_highlight(name, content, highlight)
                    yield {"type": "tool_result", "scope": scope,
                           "name": name, "content": content[:4000]}
                elif type(m).__name__ == "AIMessage" and getattr(m, "content", None) \
                        and not getattr(m, "tool_calls", None):
                    final = m.content
    yield {"type": "answer", "content": final, "highlight": sorted(highlight)}
