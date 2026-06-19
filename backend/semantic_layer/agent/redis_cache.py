"""Redis-backed query cache: exact-match, shared across workers.

Implements the same interface as QueryCache (get_exact/get_semantic/put) so the
controller is backend-agnostic. Values are JSON; keys are TTL'd via SETEX. Semantic
lookup is not supported here (it needs a Redis vector index / RediSearch — a follow-up);
get_semantic returns None so the controller cleanly falls through to compute."""

import json

from semantic_layer.agent.cache import _normalize

_PREFIX = "qcache:"


class RedisQueryCache:
    def __init__(self, client, ttl_seconds: int):
        self._r = client
        self._ttl = ttl_seconds

    def get_exact(self, question: str) -> object | None:
        raw = self._r.get(_PREFIX + _normalize(question))
        return json.loads(raw) if raw is not None else None

    def get_semantic(self, embedding: list[float], threshold: float) -> object | None:
        return None  # cross-worker semantic needs a vector index (RediSearch) — follow-up

    def put(self, question: str, answer: object, embedding: list[float] | None = None) -> None:
        self._r.setex(_PREFIX + _normalize(question), self._ttl, json.dumps(answer, default=str))
