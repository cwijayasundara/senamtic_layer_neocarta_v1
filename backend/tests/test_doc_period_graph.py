"""Graph-bound tests for the fiscal-period layer: :Period nodes, the
Document-[:COVERS_PERIOD]->Period edge, and the periods_for_documents agent tool."""

import json

import pytest

from semantic_layer.config import settings
from semantic_layer.graph.client import reset_graph
from semantic_layer.ingest.doc_loader import load_document
from semantic_layer.ingest.doc_graph import link_document_period
from semantic_layer.ingest.period_indexer import index_periods


def _doc(driver, doc_id):
    load_document(driver, {
        "doc_id": doc_id, "title": doc_id, "path": f"/tmp/{doc_id}.pdf", "num_pages": 1,
        "chunks": [{"chunk_id": f"{doc_id}:chunk:0", "doc_id": doc_id, "ordinal": 0,
                    "text": "Blackwell drove growth."}],
    })


@pytest.mark.neo4j
@pytest.mark.postgres
def test_index_periods_creates_sql_backed_period_nodes(neo4j_driver, postgres_dsn):
    reset_graph(neo4j_driver)
    n = index_periods(neo4j_driver)
    assert n >= 8
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        rec = s.run(
            "MATCH (p:Period {key:'FY2025-Q1'}) "
            "RETURN p.fiscal_year AS fy, p.quarter AS q, p.fiscal_period_id AS fpid"
        ).single()
    assert rec["fy"] == 2025 and rec["q"] == "Q1"
    assert rec["fpid"] is not None  # SQL-backed → scopable


@pytest.mark.neo4j
def test_link_document_period_creates_covers_period_edge(neo4j_driver):
    reset_graph(neo4j_driver)
    _doc(neo4j_driver, "doc:pr")
    link_document_period(neo4j_driver, "doc:pr", {"fiscal_year": 2027, "quarter": "Q1"})
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        rec = s.run(
            "MATCH (:Document {id:'doc:pr'})-[:COVERS_PERIOD]->(p:Period) "
            "RETURN p.key AS key, p.fiscal_year AS fy, p.quarter AS q"
        ).single()
    assert rec["key"] == "FY2027-Q1" and rec["fy"] == 2027 and rec["q"] == "Q1"


@pytest.mark.neo4j
def test_link_document_period_merges_onto_sql_backed_node(neo4j_driver):
    # A doc-covered period that matches an SQL-backed period must attach to the SAME
    # node and preserve its fiscal_period_id (no duplicate, no clobber).
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run("MERGE (p:Period {key:'FY2025-Q1'}) "
              "SET p.fiscal_year=2025, p.quarter='Q1', p.fiscal_period_id=5")
    _doc(neo4j_driver, "doc:pr")
    link_document_period(neo4j_driver, "doc:pr", {"fiscal_year": 2025, "quarter": "Q1"})
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        count = s.run("MATCH (p:Period {key:'FY2025-Q1'}) RETURN count(p) AS c").single()["c"]
        fpid = s.run("MATCH (p:Period {key:'FY2025-Q1'}) RETURN p.fiscal_period_id AS f").single()["f"]
    assert count == 1
    assert fpid == 5


@pytest.mark.neo4j
def test_periods_for_documents_reports_sql_availability(neo4j_driver):
    from semantic_layer.agent.graph_tools import periods_for_documents
    reset_graph(neo4j_driver)
    with neo4j_driver.session(database=settings.neo4j_database) as s:
        s.run("MERGE (p:Period {key:'FY2025-Q1'}) "
              "SET p.fiscal_year=2025, p.quarter='Q1', p.fiscal_period_id=5")
    _doc(neo4j_driver, "doc:sql")
    _doc(neo4j_driver, "doc:nosql")
    link_document_period(neo4j_driver, "doc:sql", {"fiscal_year": 2025, "quarter": "Q1"})
    link_document_period(neo4j_driver, "doc:nosql", {"fiscal_year": 2099, "quarter": "Q4"})

    out = {r["doc_id"]: r for r in
           json.loads(periods_for_documents.invoke({"doc_ids": ["doc:sql", "doc:nosql"]}))}
    assert out["doc:sql"]["fiscal_year"] == 2025 and out["doc:sql"]["sql_available"] is True
    assert out["doc:nosql"]["sql_available"] is False
