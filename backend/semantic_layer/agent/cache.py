"""In-process query cache: exact-match (normalized question) plus semantic
(embedding cosine) lookup, LRU-bounded with TTL. A Redis-backed variant (Task C3)
is the production path for multi-worker deploys; this default suits a single worker."""

import threading
import time
from collections import OrderedDict

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_openai_client


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
        self._lock = threading.Lock()
        # key -> {"answer": <value>, "embedding": list[float] | None, "ts": float}
        self._store: "OrderedDict[str, dict]" = OrderedDict()

    def _fresh(self, entry: dict) -> bool:
        return (self._now() - entry["ts"]) <= self._ttl

    def get_exact(self, question: str) -> object | None:
        key = _normalize(question)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if not self._fresh(entry):
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return entry["answer"]

    def get_semantic(self, embedding: list[float], threshold: float) -> object | None:
        with self._lock:
            best, best_sim, best_key = None, threshold, None
            for key, entry in list(self._store.items()):
                if not self._fresh(entry):
                    self._store.pop(key, None)
                    continue
                emb = entry.get("embedding")
                if emb is None:
                    continue
                sim = cosine(embedding, emb)
                if sim >= best_sim:
                    best, best_sim, best_key = entry, sim, key
            if best_key is not None:
                self._store.move_to_end(best_key)  # promote winner to MRU position
            return best["answer"] if best else None

    def put(self, question: str, answer: object, embedding: list[float] | None = None) -> None:
        key = _normalize(question)
        with self._lock:
            self._store[key] = {"answer": answer, "embedding": embedding, "ts": self._now()}
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)   # evict oldest


def embed_query(question: str) -> list[float]:
    """Embed a question with the configured embedding model for semantic cache lookup."""
    client = get_openai_client()
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=[question],
        dimensions=settings.embedding_dimensions,
    )
    return resp.data[0].embedding


def _redis_client_from_url(url: str):
    import redis  # lazy: memory backend must not require redis installed
    return redis.from_url(url)


def build_query_cache():
    """Select the cache backend from settings (memory default; redis for multi-worker)."""
    if settings.cache_backend == "redis":
        from semantic_layer.agent.redis_cache import RedisQueryCache
        return RedisQueryCache(_redis_client_from_url(settings.redis_url), settings.cache_ttl_seconds)
    return QueryCache(settings.cache_max_entries, settings.cache_ttl_seconds)


query_cache = build_query_cache()
