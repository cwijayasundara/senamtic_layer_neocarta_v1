"""API-key auth (and rate limiting) for the web API. Disabled by default."""

import threading
import time

from fastapi import Header, HTTPException, Request

from semantic_layer.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce X-API-Key against the configured allow-list.
    No-op when no keys are configured (dev default)."""
    allowed = settings.api_key_list
    if not allowed:
        return
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class RateLimiter:
    """Fixed-window per-key limiter (in-process; for multi-worker use a shared store)."""

    def __init__(self, per_min: int, now=time.monotonic, max_keys: int = 100_000):
        self._per_min = per_min
        self._now = now
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._now()
            # Bound memory: when the map is full, drop fully-expired windows so a flood
            # of distinct keys cannot grow _hits without limit (DoS guard).
            if len(self._hits) >= self._max_keys:
                self._hits = {k: v for k, v in self._hits.items() if now - v[0] < 60}
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
    """FastAPI dependency: throttle per client. No-op when disabled.

    Bucket key: when auth is ENABLED, x_api_key was already validated by require_api_key
    (it runs first), so we key on the validated key. When auth is DISABLED we key on the
    client host ONLY — never the client-controlled X-API-Key header, which an attacker
    could rotate to mint unlimited buckets and bypass the limit.
    NOTE (reverse proxy): request.client.host is the immediate peer; behind a load
    balancer this collapses all clients to the proxy IP, so size the limit accordingly
    (or add trusted-proxy X-Forwarded-For parsing — follow-up) before relying on per-IP."""
    if settings.rate_limit_per_min <= 0:
        return
    if settings.api_key_list:
        key = f"key:{x_api_key}"   # validated by require_api_key (runs before this dep)
    else:
        key = f"ip:{request.client.host if request.client else 'anon'}"
    if not _rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
