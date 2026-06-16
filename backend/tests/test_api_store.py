from semantic_layer.apis import store
from semantic_layer.apis.models import Account, Ticket, InventoryItem, UsageRecord


def test_store_returns_cached_identical_objects():
    assert store.crm_data() is store.crm_data()  # lru_cache returns same object


def test_models_validate_store_rows():
    Account(**store.crm_data()["accounts"][0])
    Ticket(**store.itsm_data()["tickets"][0])
    InventoryItem(**store.partner_data()["inventory"][0])
    UsageRecord(**store.dgx_data()["usage"][0])
