"""Support/ITSM mock API: tickets and GPU RMAs (ServiceNow-like)."""

from fastapi import FastAPI, HTTPException

from semantic_layer.apis.models import Ticket, RMA
from semantic_layer.apis.store import itsm_data

itsm_app = FastAPI(
    title="NVIDIA Support / ITSM API",
    version="1.0.0",
    description="Support tickets, severities, SLAs, and GPU RMAs.",
)


@itsm_app.get("/tickets", response_model=list[Ticket])
def list_tickets(
    severity: str | None = None,
    status: str | None = None,
    account_id: int | None = None,
):
    rows = itsm_data()["tickets"]
    if severity:
        rows = [t for t in rows if t["severity"].lower() == severity.lower()]
    if status:
        rows = [t for t in rows if t["status"].lower() == status.lower()]
    if account_id is not None:
        rows = [t for t in rows if t["account_id"] == account_id]
    return rows


@itsm_app.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: int):
    for t in itsm_data()["tickets"]:
        if t["ticket_id"] == ticket_id:
            return t
    raise HTTPException(status_code=404, detail="ticket not found")


@itsm_app.get("/rma", response_model=list[RMA])
def list_rmas(status: str | None = None):
    rows = itsm_data()["rmas"]
    if status:
        rows = [r for r in rows if r["status"].lower() == status.lower()]
    return rows
