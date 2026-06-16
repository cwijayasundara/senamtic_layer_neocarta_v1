"""CRM mock API: accounts, contacts, and sales opportunities (Salesforce-like)."""

from fastapi import FastAPI, HTTPException

from semantic_layer.apis.models import Account, Contact, Opportunity
from semantic_layer.apis.store import crm_data

crm_app = FastAPI(
    title="NVIDIA CRM API",
    version="1.0.0",
    description="Accounts, contacts, and sales opportunities.",
)


@crm_app.get("/accounts", response_model=list[Account])
def list_accounts(region: str | None = None, industry: str | None = None):
    rows = crm_data()["accounts"]
    if region:
        rows = [a for a in rows if a["region"] == region]
    if industry:
        rows = [a for a in rows if a["industry"] == industry]
    return rows


@crm_app.get("/accounts/{account_id}", response_model=Account)
def get_account(account_id: int):
    for a in crm_data()["accounts"]:
        if a["account_id"] == account_id:
            return a
    raise HTTPException(status_code=404, detail="account not found")


@crm_app.get("/opportunities", response_model=list[Opportunity])
def list_opportunities(stage: str | None = None, account_id: int | None = None):
    rows = crm_data()["opportunities"]
    if stage:
        rows = [o for o in rows if o["stage"] == stage]
    if account_id is not None:
        rows = [o for o in rows if o["account_id"] == account_id]
    return rows


@crm_app.get("/contacts", response_model=list[Contact])
def list_contacts(account_id: int | None = None):
    rows = crm_data()["contacts"]
    if account_id is not None:
        rows = [c for c in rows if c["account_id"] == account_id]
    return rows
