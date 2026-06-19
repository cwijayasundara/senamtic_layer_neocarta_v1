# Production-Scaling P2-Infra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the infrastructure primitives that make the semantic layer deployable horizontally and safely — a Redis-backed shared query cache for multi-worker deploys, API-key authentication + rate limiting on the public endpoints, and a blocking-based entity-resolution step that replaces the O(entities × values) bridge so document→catalog linking scales.

**Architecture:** Three independent features. (A) A `RedisQueryCache` implementing the existing cache interface (exact-match, cross-worker) selected by `settings.cache_backend`; the in-process `QueryCache` stays the default. (B) A FastAPI `require_api_key` dependency + in-process rate limiter guard `/chat` and `/graph`, both no-ops at the dev default (no keys configured). (C) `bridge_entities_to_values` swaps its nested-loop cross-product for token-blocking candidate generation — same edges, near-linear instead of quadratic.

**Tech Stack:** Python ≥3.11, Redis (`redis` client, `fakeredis` for tests), FastAPI, Neo4j, pydantic-settings, pytest.

## Global Constraints

- **Python:** `requires-python = ">=3.11"`. New runtime deps go in `backend/pyproject.toml` `[project].dependencies`; test-only deps in `[project.optional-dependencies].dev`.
- **Tests run from `backend/` with the project venv:** `backend/.venv/bin/python -m pytest`. The Makefile's bare `python` is NOT on PATH. `pythonpath = ["."]`.
- **Test markers (`backend/pyproject.toml`):** `neo4j`, `postgres`, `openai`, and a NEW `redis` marker (added in A1). Marker-gated tests MUST take the matching fixture so they SKIP when the service is absent. Pure-logic tests take no fixture.
- **Backward compatibility (critical):** every feature is a no-op at default settings — `cache_backend="memory"` (unchanged in-process cache), `api_keys=""` (auth disabled), and the ER change produces the identical edge set. Existing tests must stay green.
- **LLM stubbing convention** (from `tests/test_legs.py`): monkeypatch module-level `get_chat_model`; never call a real model in a non-`openai` test.
- **Graph writes use `MERGE`.**
- **⚠️ Running the full backend pytest suite wipes `Chunk.embedding`.** After a DB-touching run, restore with `backend/.venv/bin/python -m semantic_layer.ingest.pipeline`.
- **TDD + frequent commits.**

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `backend/pyproject.toml` (modify) | `redis` dep, `fakeredis` dev dep, `redis` marker | A1 |
| `docker-compose.yml` (modify) | redis service | A1 |
| `backend/semantic_layer/config.py` (modify) | `cache_backend`, `redis_url`, `api_keys`, `rate_limit_per_min` | A1, A2, B1, B2 |
| `backend/semantic_layer/agent/redis_cache.py` (create) | `RedisQueryCache` | A1 |
| `backend/semantic_layer/agent/cache.py` (modify) | `build_query_cache()` factory | A2 |
| `backend/semantic_layer/web/auth.py` (create) | `require_api_key`, `RateLimiter`, `rate_limit` dep | B1, B2 |
| `backend/semantic_layer/web/app.py` (modify) | apply auth + rate-limit deps to routes | B1, B2 |
| `backend/semantic_layer/ingest/doc_graph.py` (modify) | `_candidate_pairs` blocking ER | C1 |
| `backend/tests/test_redis_cache.py` (create) | RedisQueryCache (fakeredis) + factory | A1, A2 |
| `backend/tests/test_web_auth.py` (create) | auth + rate limit | B1, B2 |
| `backend/tests/test_entity_resolution.py` (create) | blocking ER equivalence | C1 |

Features A, B, C are mutually independent. Within each, do tasks in order.

---

# Feature A — Redis-Backed Shared Cache

**Why:** The P0 cache and P1 concurrency gate are per-process (`cache.py:87` instantiates one `QueryCache` per worker). Under multiple uvicorn/gunicorn workers, repeated queries miss the cache on every worker. A Redis-backed exact-match cache shares hits across workers. (Cross-worker *semantic* caching needs a Redis vector index (RediSearch) — deferred; see the end of this plan.)

---

### Task A1: `RedisQueryCache` + deps + redis service

**Files:**
- Modify: `backend/pyproject.toml`, `docker-compose.yml`, `backend/semantic_layer/config.py`
- Create: `backend/semantic_layer/agent/redis_cache.py`
- Test: `backend/tests/test_redis_cache.py`

**Interfaces:**
- Produces: `RedisQueryCache(client, ttl_seconds)` with `get_exact(question) -> object | None`, `get_semantic(embedding, threshold) -> None` (always None — see note), `put(question, answer, embedding=None) -> None`. Same interface as `QueryCache`; values are JSON-serialized; keys are `qcache:<normalized question>`; TTL via `SETEX`.

- [ ] **Step 1: Add deps, marker, and the redis service**

In `backend/pyproject.toml` `[project].dependencies` add:
```toml
    "redis>=5",
```
In `[project.optional-dependencies].dev` add `fakeredis`:
```toml
dev = ["pytest>=8.0", "httpx>=0.27,<0.29", "fakeredis>=2.20"]
```
In `[tool.pytest.ini_options].markers` add:
```toml
    "redis: tests requiring a reachable Redis service",
```
Install into the venv:
Run: `cd backend && .venv/bin/python -m pip install "redis>=5" "fakeredis>=2.20"`
Expected: both install.

In `docker-compose.yml`, add a redis service (under `services:`) and leave volumes as-is:
```yaml
  redis:
    image: redis:7
    container_name: neocarta-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10
```

- [ ] **Step 2: Add settings**

In `backend/semantic_layer/config.py`, after the cache settings (`cache_similarity_threshold`):
```python
    # Query cache backend: "memory" (in-process, default) or "redis" (shared across
    # workers, exact-match only — cross-worker semantic needs a vector index, see plan).
    cache_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
```

- [ ] **Step 3: Write the failing tests**

```python
# backend/tests/test_redis_cache.py
import fakeredis

from semantic_layer.agent.redis_cache import RedisQueryCache


def _cache():
    return RedisQueryCache(fakeredis.FakeStrictRedis(), ttl_seconds=1000)


def test_put_then_get_exact_roundtrip_normalized():
    c = _cache()
    c.put("What is revenue?", [{"type": "answer", "content": "42"}])
    assert c.get_exact("  what   IS revenue? ") == [{"type": "answer", "content": "42"}]


def test_get_exact_miss_returns_none():
    assert _cache().get_exact("nothing here") is None


def test_get_semantic_is_none_in_redis_backend():
    c = _cache()
    c.put("q", [{"type": "answer", "content": "x"}], embedding=[0.1, 0.2])
    assert c.get_semantic([0.1, 0.2], 0.9) is None   # cross-worker semantic deferred


def test_put_sets_ttl():
    client = fakeredis.FakeStrictRedis()
    RedisQueryCache(client, ttl_seconds=123).put("q", [{"type": "answer", "content": "x"}])
    # one key written, with a TTL close to the configured ttl
    keys = client.keys("qcache:*")
    assert len(keys) == 1
    assert 0 < client.ttl(keys[0]) <= 123
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_redis_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'semantic_layer.agent.redis_cache'`.

- [ ] **Step 5: Implement**

```python
# backend/semantic_layer/agent/redis_cache.py
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
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_redis_cache.py -v`
Expected: PASS (4 passed). No real Redis needed — fakeredis is in-memory.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml docker-compose.yml backend/semantic_layer/config.py backend/semantic_layer/agent/redis_cache.py backend/tests/test_redis_cache.py
git commit -m "feat(cache): RedisQueryCache (exact-match, cross-worker) + redis service"
```

---

### Task A2: Backend factory + wire the singleton

**Files:**
- Modify: `backend/semantic_layer/agent/cache.py`
- Test: `backend/tests/test_redis_cache.py`

**Interfaces:**
- Produces: `build_query_cache()` in `cache.py` — returns a `RedisQueryCache` (lazy `redis.from_url(settings.redis_url)`) when `settings.cache_backend == "redis"`, else the in-process `QueryCache`. The module singleton `query_cache` is built via this factory. The controller import (`from semantic_layer.agent.cache import query_cache`) is unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# append to backend/tests/test_redis_cache.py
import fakeredis as _fr

from semantic_layer.agent import cache as cache_mod
from semantic_layer.agent.cache import build_query_cache, QueryCache
from semantic_layer.agent.redis_cache import RedisQueryCache


def test_factory_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_backend", "memory", raising=False)
    assert isinstance(build_query_cache(), QueryCache)


def test_factory_builds_redis_when_configured(monkeypatch):
    monkeypatch.setattr(cache_mod.settings, "cache_backend", "redis", raising=False)
    monkeypatch.setattr(cache_mod, "_redis_client_from_url", lambda url: _fr.FakeStrictRedis())
    c = build_query_cache()
    assert isinstance(c, RedisQueryCache)
    c.put("q", [{"type": "answer", "content": "ok"}])
    assert c.get_exact("q") == [{"type": "answer", "content": "ok"}]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_redis_cache.py::test_factory_builds_redis_when_configured -v`
Expected: FAIL — `cache.py` has no `build_query_cache` / `_redis_client_from_url`.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/agent/cache.py`, replace the final line (`query_cache = QueryCache(...)`) with the factory and a seam for the redis client:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_redis_cache.py tests/test_query_cache.py tests/test_controller.py -v`
Expected: PASS — factory tests green; existing cache/controller tests stay green (default is still `QueryCache`).

- [ ] **Step 5: (Optional) redis-marker integration test**

```python
# append to backend/tests/test_redis_cache.py
import pytest


@pytest.mark.redis
def test_real_redis_roundtrip():
    import redis as _redis
    from semantic_layer.config import settings
    try:
        client = _redis.from_url(settings.redis_url)
        client.ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis not available: {exc}")
    c = RedisQueryCache(client, ttl_seconds=60)
    c.put("integration q", [{"type": "answer", "content": "real"}])
    assert c.get_exact("integration q") == [{"type": "answer", "content": "real"}]
    client.delete("qcache:integration q")
```

Run: `cd backend && .venv/bin/python -m pytest tests/test_redis_cache.py -v`
Expected: factory + unit tests PASS; the redis-marked test runs if `docker compose up redis` is up, else SKIPs.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/agent/cache.py backend/tests/test_redis_cache.py
git commit -m "feat(cache): backend factory (memory|redis) for the query cache singleton"
```

---

# Feature B — API Authentication + Rate Limiting

**Why:** `/chat` and `/graph` are unauthenticated and unthrottled (`web/app.py`) — unacceptable before any external exposure, and `/chat` is expensive (multi-LLM). Add API-key auth and a per-key rate limit, both **disabled by default** (no keys configured) so dev/tests are unaffected.

---

### Task B1: API-key authentication dependency

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Create: `backend/semantic_layer/web/auth.py`
- Modify: `backend/semantic_layer/web/app.py`
- Test: `backend/tests/test_web_auth.py`

**Interfaces:**
- Produces: `require_api_key(x_api_key: str | None = Header(default=None)) -> None` in `auth.py` — when `settings.api_key_list` is empty, it's a no-op (auth disabled); otherwise it raises `HTTPException(401)` unless `x_api_key` is in the allow-list. `settings.api_key_list` parses `settings.api_keys` (CSV). Applied to `/chat` and `/graph` via `dependencies=[Depends(require_api_key)]`; `/health` stays open.

- [ ] **Step 1: Add settings**

In `backend/semantic_layer/config.py`:
```python
    # API auth: comma-separated allowed keys (sent as the X-API-Key header). Empty =
    # auth DISABLED (dev default). Set to enable; clients must present a listed key.
    api_keys: str = ""
```
and a property (next to `api_source_list`):
```python
    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]
```

- [ ] **Step 2: Write the failing tests**

```python
# backend/tests/test_web_auth.py
import pytest
from fastapi import HTTPException

from semantic_layer.web import auth as auth_mod


def test_require_api_key_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "", raising=False)
    assert auth_mod.require_api_key(x_api_key=None) is None   # disabled -> allow


def test_require_api_key_rejects_missing_and_bad(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "good-key,other", raising=False)
    with pytest.raises(HTTPException) as e1:
        auth_mod.require_api_key(x_api_key=None)
    assert e1.value.status_code == 401
    with pytest.raises(HTTPException) as e2:
        auth_mod.require_api_key(x_api_key="nope")
    assert e2.value.status_code == 401


def test_require_api_key_accepts_listed_key(monkeypatch):
    monkeypatch.setattr(auth_mod.settings, "api_keys", "good-key", raising=False)
    assert auth_mod.require_api_key(x_api_key="good-key") is None


def test_graph_endpoint_enforces_key(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod
    monkeypatch.setattr(app_mod.settings, "api_keys", "secret", raising=False)
    monkeypatch.setattr(app_mod, "get_schema_graph",
                        lambda source=None, max_chunks=None: {"nodes": [], "edges": [], "truncated": False})
    client = TestClient(app_mod.app)
    assert client.get("/graph").status_code == 401
    assert client.get("/graph", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200   # health stays open
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_auth.py -v`
Expected: FAIL — `semantic_layer.web.auth` does not exist.

- [ ] **Step 4: Implement**

Create `backend/semantic_layer/web/auth.py`:
```python
"""API-key auth (and rate limiting) for the web API. Disabled by default."""

from fastapi import Header, HTTPException

from semantic_layer.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce X-API-Key against the configured allow-list.
    No-op when no keys are configured (dev default)."""
    allowed = settings.api_key_list
    if not allowed:
        return
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
```

In `backend/semantic_layer/web/app.py`, import the dependency and apply it to the protected routes:
```python
from fastapi import FastAPI, Depends
...
from semantic_layer.web.auth import require_api_key
```
Change the two route decorators:
```python
@app.get("/graph", dependencies=[Depends(require_api_key)])
def graph(source: str | None = None, max_chunks: int | None = None):
    ...

@app.post("/chat", dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    ...
```
(Leave `/health` and `/sources` as they are, or add the dep to `/sources` too if you want it protected — keep `/health` open for healthchecks.)

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_auth.py tests/test_web_chat_events.py tests/test_web_graph_api.py tests/test_api_app.py -v`
Expected: new auth tests PASS; existing web tests stay green (they don't set `api_keys`, so auth is disabled and behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/web/auth.py backend/semantic_layer/web/app.py backend/tests/test_web_auth.py
git commit -m "feat(web): API-key auth on /chat and /graph (disabled by default)"
```

---

### Task B2: Per-key rate limiting on `/chat`

**Files:**
- Modify: `backend/semantic_layer/config.py`
- Modify: `backend/semantic_layer/web/auth.py`
- Modify: `backend/semantic_layer/web/app.py`
- Test: `backend/tests/test_web_auth.py`

**Interfaces:**
- Produces: `RateLimiter(per_min, now=time.monotonic)` with `allow(key: str) -> bool` (fixed 60s window per key) in `auth.py`; a module `_rate_limiter` built from `settings.rate_limit_per_min`; a `rate_limit(x_api_key, request)` FastAPI dependency raising `HTTPException(429)` when over the limit. Disabled when `settings.rate_limit_per_min <= 0`. Keyed by API key, falling back to client host. Applied to `/chat`.

- [ ] **Step 1: Add setting**

In `backend/semantic_layer/config.py`:
```python
    # Per-client requests/minute on /chat (keyed by API key, else client IP).
    # 0 disables rate limiting (dev default).
    rate_limit_per_min: int = 0
```

- [ ] **Step 2: Write the failing tests**

```python
# append to backend/tests/test_web_auth.py
def test_rate_limiter_allows_then_blocks():
    clock = {"t": 1000.0}
    rl = auth_mod.RateLimiter(per_min=2, now=lambda: clock["t"])
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is False        # 3rd in the window blocked
    clock["t"] = 1061.0                  # next window
    assert rl.allow("k") is True


def test_rate_limiter_is_per_key():
    rl = auth_mod.RateLimiter(per_min=1, now=lambda: 5.0)
    assert rl.allow("a") is True
    assert rl.allow("b") is True         # different key, own budget
    assert rl.allow("a") is False
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_auth.py::test_rate_limiter_allows_then_blocks -v`
Expected: FAIL — `auth_mod.RateLimiter` does not exist.

- [ ] **Step 4: Implement**

In `backend/semantic_layer/web/auth.py`, add:
```python
import threading
import time

from fastapi import Request


class RateLimiter:
    """Fixed-window per-key limiter (in-process; for multi-worker use a shared store)."""

    def __init__(self, per_min: int, now=time.monotonic):
        self._per_min = per_min
        self._now = now
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._now()
            start, count = self._hits.get(key, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            if count >= self._per_min:
                self._hits[key] = (start, count)
                return False
            self._hits[key] = (start, count + 1)
            return True


_rate_limiter = RateLimiter(settings.rate_limit_per_min)


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: throttle per API key (else client host). No-op when disabled."""
    if settings.rate_limit_per_min <= 0:
        return
    key = x_api_key or (request.client.host if request.client else "anon")
    if not _rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
```

In `backend/semantic_layer/web/app.py`, add `rate_limit` to the `/chat` dependencies:
```python
from semantic_layer.web.auth import require_api_key, rate_limit
...
@app.post("/chat", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def chat(req: ChatRequest):
    ...
```

- [ ] **Step 5: Add an endpoint rate-limit test**

```python
# append to backend/tests/test_web_auth.py
def test_chat_endpoint_rate_limits(monkeypatch):
    from fastapi.testclient import TestClient
    from semantic_layer.web import app as app_mod
    from semantic_layer.web import auth as auth_module
    monkeypatch.setattr(app_mod.settings, "api_keys", "", raising=False)        # auth off
    monkeypatch.setattr(auth_module.settings, "rate_limit_per_min", 1, raising=False)
    monkeypatch.setattr(auth_module, "_rate_limiter", auth_module.RateLimiter(per_min=1, now=lambda: 0.0))
    # avoid running the real agent: stub the SSE source to a trivial generator
    monkeypatch.setattr(app_mod, "stream_chat_events",
                        lambda q: iter([{"type": "answer", "content": "x"}]))
    client = TestClient(app_mod.app)
    assert client.post("/chat", json={"question": "q"}).status_code == 200
    assert client.post("/chat", json={"question": "q"}).status_code == 429
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_web_auth.py tests/test_web_chat_events.py -v`
Expected: all auth + rate-limit tests PASS; existing chat-events test green (rate limit defaults to 0 = disabled).

- [ ] **Step 7: Commit**

```bash
git add backend/semantic_layer/config.py backend/semantic_layer/web/auth.py backend/semantic_layer/web/app.py backend/tests/test_web_auth.py
git commit -m "feat(web): per-key rate limiting on /chat (disabled by default)"
```

---

# Feature C — Scalable Entity Resolution

**Why:** `bridge_entities_to_values` (`doc_graph.py:130-134`) builds its `Entity -> Value` bridge with a full nested loop — `for en in entities for vn in values` — i.e. O(entities × values). At thousands of documents (tens of thousands of entities) × thousands of catalog values this is quadratic and dominates ingest. Replace it with token blocking: index values by token, generate candidates only for entities sharing a token, then apply the existing `_token_match`. Same edges, near-linear.

---

### Task C1: Token-blocking candidate generation

**Files:**
- Modify: `backend/semantic_layer/ingest/doc_graph.py`
- Test: `backend/tests/test_entity_resolution.py`

**Interfaces:**
- Produces: `_candidate_pairs(entities: list[str], values: list[str]) -> list[dict]` — returns the SAME set of `{"e", "v"}` pairs as the brute-force `[{"e":en,"v":vn} for en in entities for vn in values if en==vn or _token_match(en,vn)]`, via token blocking. `bridge_entities_to_values` uses it instead of the nested loop.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_entity_resolution.py
from semantic_layer.ingest.doc_graph import _candidate_pairs, _token_match


def _bruteforce(entities, values):
    return {(en, vn) for en in entities for vn in values if en == vn or _token_match(en, vn)}


def test_candidate_pairs_equals_bruteforce_on_sample():
    entities = ["nvidia blackwell gpus", "data center revenue", "jensen huang",
                "emea cloud customers", "blackwell"]
    values = ["blackwell", "data center", "cloud service provider", "emea", "hopper",
              "data center", "q1"]
    got = {(p["e"], p["v"]) for p in _candidate_pairs(entities, values)}
    assert got == _bruteforce(entities, values)


def test_candidate_pairs_no_duplicates():
    pairs = _candidate_pairs(["blackwell blackwell", "blackwell"], ["blackwell"])
    keys = [(p["e"], p["v"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_candidate_pairs_empty_inputs():
    assert _candidate_pairs([], ["x"]) == []
    assert _candidate_pairs(["x"], []) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_resolution.py -v`
Expected: FAIL — `_candidate_pairs` does not exist.

- [ ] **Step 3: Implement**

In `backend/semantic_layer/ingest/doc_graph.py`, add (above `bridge_entities_to_values`):
```python
from collections import defaultdict


def _candidate_pairs(entities: list[str], values: list[str]) -> list[dict]:
    """Token-blocked Entity->Value candidate pairs, equivalent to the brute-force
    cross-product filtered by (en == vn or _token_match) but near-linear: a value can
    only match an entity that shares one of the value's tokens, so we index values by
    token and probe only the candidates per entity."""
    by_token: dict[str, list[str]] = defaultdict(list)
    for vn in values:
        for tok in set(vn.split()):
            by_token[tok].append(vn)
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for en in entities:
        candidates: set[str] = set()
        for tok in set(en.split()):
            candidates.update(by_token.get(tok, ()))
        for vn in candidates:
            if (en, vn) in seen:
                continue
            if en == vn or _token_match(en, vn):
                seen.add((en, vn))
                pairs.append({"e": en, "v": vn})
    return pairs
```

In `bridge_entities_to_values`, replace the nested-loop `pairs = [...]` (the comprehension at lines ~130-134) with:
```python
        pairs = _candidate_pairs(entities, values)
```
(Leave the surrounding session reads, the `MERGE` UNWIND write, and the count return unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_resolution.py tests/test_doc_graph_bridge.py -v`
Expected: equivalence tests PASS; the existing `test_doc_graph_bridge.py` stays green (or SKIPs if neo4j-marked without a DB) — the produced edges are identical. If a DB-touching run wiped embeddings, restore with `make ingest`.

- [ ] **Step 5: Commit**

```bash
git add backend/semantic_layer/ingest/doc_graph.py backend/tests/test_entity_resolution.py
git commit -m "feat(ingest): token-blocking entity resolution (drops O(E*V) bridge)"
```

---

## Deferred to a separate plan (P2-infra-ops) — and why

These remaining scale-out items are provisioning/ops or large-design efforts, not application code amenable to TDD here:

1. **Distributed / bulk ingestion** — `neo4j-admin database import full` is a CLI run against a stopped/empty DB with format-specific CSVs; correctness needs live iteration against the container. A code-able first step (a `bulk_export` that emits neo4j-admin-format CSVs) plus the import runbook + a streaming/CDC incremental pipeline is its own plan.
2. **Horizontal Neo4j** (clustering / Infinigraph sharding, read replicas) — Neo4j Enterprise + deployment topology; essentially no application code, all ops.
3. **True multi-tenant data isolation** — partitioning the graph/SQL/APIs per tenant and threading a tenant id through every query/tool is a cross-cutting design spanning the whole stack; auth (Feature B) is the prerequisite, not the whole thing.
4. **Cross-worker semantic cache (RediSearch)** — a Redis vector index for `get_semantic` in `RedisQueryCache`; needs the redis-stack module and an index-management design. Feature A delivers cross-worker *exact-match* now.
5. **Distributed rate limiting** — the Feature B limiter is per-process; a multi-worker deploy needs the counter in Redis (composes with Feature A's client).

---

## Self-Review

**Spec coverage** (against the P2-infra gaps):
- ✅ Redis shared cache (multi-worker) — Feature A (exact-match; semantic via RediSearch deferred).
- ✅ Auth / rate-limiting — Feature B (API-key auth + per-key rate limit, both default-off).
- ✅ Entity resolution at scale — Feature C (token blocking, edge-equivalent, near-linear).
- ⏸️ Bulk/distributed ingest, horizontal Neo4j, multi-tenant data isolation, RediSearch semantic cache, distributed rate limiting — deferred to P2-infra-ops with rationale.

**Placeholder scan:** No TBDs; every code step has complete code; every run step states the venv command and expected result.

**Type consistency:**
- `RedisQueryCache(client, ttl_seconds)` with `get_exact/get_semantic/put` matches the `QueryCache` interface the controller calls; `build_query_cache() -> QueryCache | RedisQueryCache` (A1↔A2); `_redis_client_from_url(url)` is the monkeypatch seam.
- `require_api_key(x_api_key=Header)` and `rate_limit(request, x_api_key=Header)` are FastAPI deps; `settings.api_key_list`, `settings.rate_limit_per_min`; `RateLimiter(per_min, now).allow(key) -> bool` (B1↔B2).
- `_candidate_pairs(entities, values) -> list[{e, v}]` consumed by `bridge_entities_to_values` (C1).

**Backward-compat:** `cache_backend` default `"memory"` (factory returns the existing `QueryCache`); `api_keys` default `""` (auth no-op); `rate_limit_per_min` default `0` (limiter no-op); Feature C produces the identical edge set (asserted against brute force). Existing suites stay green.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-production-scaling-p2-infra.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
