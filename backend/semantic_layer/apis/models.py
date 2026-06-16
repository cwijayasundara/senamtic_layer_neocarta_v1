"""Pydantic response models for the mock enterprise APIs.

These drive each sub-app's OpenAPI schema, which Plan 3's NeoCarta API
extractor introspects to build virtual-table/column metadata nodes.
"""

from pydantic import BaseModel


class Account(BaseModel):
    account_id: int
    name: str
    industry: str
    region: str
    tier: str


class Contact(BaseModel):
    contact_id: int
    account_id: int
    name: str
    title: str
    email: str


class Opportunity(BaseModel):
    opportunity_id: int
    account_id: int
    name: str
    stage: str
    amount: int
    product_line: str
    close_date: str


class Ticket(BaseModel):
    ticket_id: int
    account_id: int
    subject: str
    severity: str
    status: str
    sla_hours: int
    product_line: str
    opened_at: str


class RMA(BaseModel):
    rma_id: int
    ticket_id: int
    product: str
    serial: str
    status: str


class Partner(BaseModel):
    partner_id: int
    name: str
    region: str
    tier: str


class InventoryItem(BaseModel):
    inventory_id: int
    partner_id: int
    product_line: str
    on_hand: int
    allocated: int
    available: int


class UsageRecord(BaseModel):
    usage_id: int
    account_id: int
    instance_type: str
    gpu_hours: float
    utilization_pct: float
    usage_date: str
