"""FastAPI web API for the semantic-layer UI."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from semantic_layer.web.graph_api import get_sources, get_schema_graph
from semantic_layer.web.events import stream_chat_events
from semantic_layer.agent.pg_pool import ensure_pool_open, get_pool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_pool_open()        # warm the Postgres pool at startup
    yield
    get_pool().close()        # release pooled connections on shutdown


app = FastAPI(title="NeoCarta-Local Web API", lifespan=lifespan)
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
