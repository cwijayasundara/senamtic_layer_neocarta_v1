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
