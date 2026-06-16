"""Partner / channel inventory mock API: distributors and stock levels."""

from fastapi import FastAPI

from semantic_layer.apis.models import Partner, InventoryItem
from semantic_layer.apis.store import partner_data

partner_app = FastAPI(
    title="NVIDIA Partner Inventory API",
    version="1.0.0",
    description="Distributor partners and channel inventory by product line.",
)


@partner_app.get("/partners", response_model=list[Partner])
def list_partners(region: str | None = None):
    rows = partner_data()["partners"]
    if region:
        rows = [p for p in rows if p["region"] == region]
    return rows


@partner_app.get("/inventory", response_model=list[InventoryItem])
def list_inventory(partner_id: int | None = None, product_line: str | None = None):
    rows = partner_data()["inventory"]
    if partner_id is not None:
        rows = [i for i in rows if i["partner_id"] == partner_id]
    if product_line:
        rows = [i for i in rows if i["product_line"] == product_line]
    return rows
