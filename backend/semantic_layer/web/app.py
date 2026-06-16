"""FastAPI web API for the semantic-layer UI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from semantic_layer.web.graph_api import get_sources, get_schema_graph

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
