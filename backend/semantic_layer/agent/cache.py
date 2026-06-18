"""In-process query cache: exact-match (normalized question) plus semantic
(embedding cosine) lookup, LRU-bounded with TTL. A Redis-backed variant (Task C3)
is the production path for multi-worker deploys; this default suits a single worker."""

import time
from collections import OrderedDict

from semantic_layer.config import settings


def _normalize(question: str) -> str:
    return " ".join((question or "").lower().split())


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class QueryCache:
    def __init__(self, max_entries: int, ttl_seconds: int, now=time.monotonic):
        self._max = max_entries
        self._ttl = ttl_seconds
        self._now = now
        # key -> {"answer": dict, "embedding": list[float] | None, "ts": float}
        self._store: "OrderedDict[str, dict]" = OrderedDict()

    def _fresh(self, entry: dict) -> bool:
        return (self._now() - entry["ts"]) <= self._ttl

    def get_exact(self, question: str) -> dict | None:
        key = _normalize(question)
        entry = self._store.get(key)
        if entry is None:
            return None
        if not self._fresh(entry):
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry["answer"]

    def get_semantic(self, embedding: list[float], threshold: float) -> dict | None:
        best, best_sim = None, threshold
        for key, entry in list(self._store.items()):
            if not self._fresh(entry):
                self._store.pop(key, None)
                continue
            emb = entry.get("embedding")
            if emb is None:
                continue
            sim = cosine(embedding, emb)
            if sim >= best_sim:
                best, best_sim = entry, sim
        return best["answer"] if best else None

    def put(self, question: str, answer: dict, embedding: list[float] | None = None) -> None:
        key = _normalize(question)
        self._store[key] = {"answer": answer, "embedding": embedding, "ts": self._now()}
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)   # evict oldest


query_cache = QueryCache(settings.cache_max_entries, settings.cache_ttl_seconds)
