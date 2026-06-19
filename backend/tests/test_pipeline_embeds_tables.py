import inspect

from semantic_layer.ingest import pipeline, embeddings


def test_llm_stages_embeds_tables_not_metadata_nodes():
    src = inspect.getsource(pipeline._run_llm_stages)
    assert "embed_tables(" in src
    assert "embed_metadata_nodes" not in src


def test_embed_metadata_nodes_removed():
    assert not hasattr(embeddings, "embed_metadata_nodes")
    assert hasattr(embeddings, "embed_tables")
