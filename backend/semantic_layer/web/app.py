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
from semantic_layer.config import settings


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
def graph(source: str | None = None, max_chunks: int | None = None):
    # Clamp a caller-supplied cap to [0, server default] so a request can only NARROW
    # the chunk layer, never exceed the server bound (resource-exhaustion guard) or go
    # negative (which would error the underlying LIMIT).
    effective = max(0, min(max_chunks, settings.graph_max_chunks)) if max_chunks is not None else None
    return get_schema_graph(source=source, max_chunks=effective)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    def event_generator():
        for event in stream_chat_events(req.question):
            yield {"event": event["type"], "data": json.dumps(event)}
    return EventSourceResponse(event_generator())
