"""FastAPI web API for the semantic-layer UI."""

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from semantic_layer.web.graph_api import get_sources, get_schema_graph
from semantic_layer.web.events import stream_chat_events

app = FastAPI(title="NeoCarta-Local Web API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    return get_sources()


@app.get("/graph")
def graph():
    return get_schema_graph()


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    def event_generator():
        for event in stream_chat_events(req.question):
            yield {"event": event["type"], "data": json.dumps(event)}
    return EventSourceResponse(event_generator())
