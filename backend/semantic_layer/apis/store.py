"""Materialize the mock-API data once (cached) from the deterministic generators."""

from functools import lru_cache

from semantic_layer.config import settings
from data.generators.api_data import (
    generate_crm,
    generate_itsm,
    generate_partner,
    generate_dgx,
)


@lru_cache
def crm_data():
    return generate_crm(settings.random_seed)


@lru_cache
def itsm_data():
    return generate_itsm(settings.random_seed)


@lru_cache
def partner_data():
    return generate_partner(settings.random_seed)


@lru_cache
def dgx_data():
    return generate_dgx(settings.random_seed)
