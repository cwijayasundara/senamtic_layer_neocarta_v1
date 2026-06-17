"""Graph-native planner: one LLM intent pass, then deterministic graph planning.

extract_intent(question) -> Intent     (one structured LLM call, planner_model)
build_plan(intent)       -> Plan dict   (pure Cypher; added in a later task)

This replaces the orchestrator's ~20 LLM discovery round-trips with a single intent
read plus a few set-based graph queries.
"""

from pydantic import BaseModel, Field

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model

_INTENT_PROMPT = (
    "You read a business question over an NVIDIA enterprise semantic layer that unifies "
    "SQL databases, REST APIs, and documents. Extract a structured intent.\n"
    "- terms: the dimension FILTER descriptors mentioned (e.g. 'EMEA','Cloud','Blackwell',"
    "'Data Center'). Split compound noun phrases into separate descriptors.\n"
    "- fact: the measure/metric in plain words (e.g. 'revenue','gpu usage','open tickets'), or null.\n"
    "- group_by: dimensions to break results down by (e.g. ['customer','quarter']).\n"
    "- fiscal_year / quarter: a fiscal scope if stated (e.g. 2025 / 'Q1'), else null.\n"
    "- needs_sql / needs_api / needs_doc: which source TYPES the question requires.\n"
    "- doc_query: what to look up in the documents, or null.\n"
    "- api_intents: enterprise-system lookups implied (e.g. ['dgx usage','open tickets'])."
)


class Intent(BaseModel):
    terms: list[str] = Field(default_factory=list)
    fact: str | None = None
    group_by: list[str] = Field(default_factory=list)
    fiscal_year: int | None = None
    quarter: str | None = None
    needs_sql: bool = True
    needs_api: bool = False
    needs_doc: bool = False
    doc_query: str | None = None
    api_intents: list[str] = Field(default_factory=list)


def extract_intent(question: str) -> Intent:
    """One structured LLM call (planner_model) -> Intent."""
    model = get_chat_model(settings.planner_model_resolved).with_structured_output(Intent)
    return model.invoke([("system", _INTENT_PROMPT), ("human", question)])
