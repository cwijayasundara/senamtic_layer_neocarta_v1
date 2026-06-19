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
