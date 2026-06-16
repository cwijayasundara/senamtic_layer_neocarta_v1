"""DGX Cloud telemetry mock API: per-account GPU usage time-series."""

from fastapi import FastAPI

from semantic_layer.apis.models import UsageRecord
from semantic_layer.apis.store import dgx_data

dgx_app = FastAPI(
    title="NVIDIA DGX Cloud Telemetry API",
    version="1.0.0",
    description="Per-account GPU-hours, utilization, and instance types.",
)


@dgx_app.get("/usage", response_model=list[UsageRecord])
def list_usage(
    account_id: int | None = None,
    instance_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = dgx_data()["usage"]
    if account_id is not None:
        rows = [u for u in rows if u["account_id"] == account_id]
    if instance_type:
        rows = [u for u in rows if u["instance_type"] == instance_type]
    if start:
        rows = [u for u in rows if u["usage_date"] >= start]
    if end:
        rows = [u for u in rows if u["usage_date"] <= end]
    return rows
