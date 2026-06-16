"""Compose the four mock enterprise APIs as mounted sub-applications.

Each API is mounted under its own path prefix so it exposes an independent
OpenAPI spec at /{prefix}/openapi.json. Run with:
    uvicorn semantic_layer.apis.app:app --port 8001
"""

from fastapi import FastAPI

from semantic_layer.apis.crm import crm_app
from semantic_layer.apis.itsm import itsm_app
from semantic_layer.apis.partner import partner_app
from semantic_layer.apis.dgx import dgx_app

app = FastAPI(title="NVIDIA Enterprise Mock APIs")


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/crm", crm_app)
app.mount("/itsm", itsm_app)
app.mount("/partner", partner_app)
app.mount("/dgx", dgx_app)
