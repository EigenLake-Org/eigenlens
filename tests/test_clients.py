"""Caller-owned SDK clients: real Qdrant reads, Weaviate mapping and lifecycle."""
from contextlib import closing
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from weaviate import WeaviateClient

import vector_tools as vt
from vector_tools.databases.qdrant import QdrantAdapter
from vector_tools.databases.weaviate import WeaviateAdapter
from test_density import fixture_data


@pytest.fixture
def client():
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("questions", vectors_config={"body": models.VectorParams(size=3, distance=models.Distance.COSINE)})
        client.upsert("questions", [models.PointStruct(id=int(row.id), vector={"body": row.vector},
            payload={"content": row.text, "theme": row.metadata["intent"], "embedding_model": "fixture-model"})
            for row in fixture_data().records])
        yield client


def test_real_qdrant_client_paginated_analysis_and_ownership(client, monkeypatch):
    monkeypatch.setattr("vector_tools.databases.qdrant.QdrantClient", MagicMock(side_effect=AssertionError("must reuse supplied client")))
    close = MagicMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    scroll = MagicMock(wraps=client.scroll)
    monkeypatch.setattr(client, "scroll", scroll)
    dataset = vt.from_client(client, "questions", text_field="content", limits=vt.Limits(batch_size=7))
    assert not scroll.called  # Construction is lazy.
    data = dataset.materialize()
    assert len(data.records) == data.read_count == 33 and data.scope == "complete_selection"
    assert scroll.call_count == 5
    assert data.source["vector_field"] == "body"
    assert data.source["embedding_model"] == "fixture-model"
    assert data.source["client_ownership"] == "caller"
    assert set(data.source).isdisjoint({"client", "url", "api_key"})
    assert not close.called
    assert client.get_collection("questions").points_count == 33
    result = vt.analyze(data, [
        {"id": "groups", "type": "cluster", "algorithm": "dbscan_search"},
        {"id": "themes", "type": "topics", "clusters_from": "groups"},
    ])
    assert len(result["groups"].rows) == 33
    assert result["themes"].versions["bertopic"]
    assert result["groups"].source == data.source
    assert len(dataset.materialize().records) == 33  # Reuse the same connection.


def test_qdrant_search_ids_filters_and_scan_budget(client):
    vector = client.retrieve("questions", [0], with_vectors=True)[0].vector["body"]
    data = vt.from_client(client, "questions", query_vector=np.array(vector), search_limit=9).materialize()
    expected = client.query_points("questions", query=vector, using="body", limit=9).points
    assert {r.id for r in data.records} == {str(p.id) for p in expected}
    assert data.scope == "retrieval_results" and data.source["mode"] == "top_k"
    assert data.source["search_limit"] == 9
    assert "query_vector" not in data.source
    assert vt.from_client(client, "questions", query_vector=vector, search_limit=9).filter(theme="card").materialize().scope == "retrieval_results"
    selected = vt.from_client(client, "questions").ids(["0", "missing"]).materialize()
    assert [r.id for r in selected.records] == ["0"]
    assert selected.missing_ids == ["missing"] and selected.read_count == 33
    assert vt.from_client(client, "questions").filter(theme="card").materialize().scope == "filtered_selection"
    a = vt.from_client(client, "questions").sample(5, seed=7).materialize()
    b = vt.from_client(client, "questions").sample(5, seed=7).materialize()
    assert a.fingerprint == b.fingerprint and a.scope == "sample"
    with pytest.raises(vt.VTError, match="max_scan_rows"):
        vt.from_client(client, "questions", limits=vt.Limits(max_scan_rows=2)).ids(["0"]).materialize()


@pytest.mark.parametrize("query", [[], [0, 0], [float("nan"), 1]])
def test_invalid_query_vectors(client, query):
    with pytest.raises(vt.VTError, match="finite, nonzero"):
        vt.from_client(client, "questions", query_vector=query)


def test_query_and_collection_validation(client):
    with pytest.raises(vt.VTError, match="dimension"):
        vt.from_client(client, "questions", query_vector=[1, 2]).materialize()
    with pytest.raises(vt.VTError, match="max_rows"):
        vt.from_client(client, "questions", query_vector=[1, 2], search_limit=11, limits=vt.Limits(max_rows=10))
    with pytest.raises(vt.VTError, match="positive integer"):
        vt.from_client(client, "questions", search_limit=True)
    with pytest.raises(vt.VTError, match="Collection name"):
        vt.from_client(client, "")
    with pytest.raises(vt.VTError, match="synchronous"):
        vt.from_client(object(), "questions")
    with pytest.raises(vt.VTError, match="vector field"):
        vt.from_client(client, "questions", vector_field="unknown").materialize()
    with pytest.raises(vt.VTError, match="Invalid"):
        vt.from_client(client, "questions", text_field="")


def weaviate_client(monkeypatch, *, field="body"):
    client = MagicMock(spec=WeaviateClient)
    collection = MagicMock()
    collection.config.get.return_value = NS(vector_config={field: object()})
    client.collections = NS(use=MagicMock(return_value=collection))
    state = {"closed": 0}
    rows = [NS(uuid=UUID(int=i + 1), vector={field: [1., float(i + 1)]},
               properties={"content": f"ATM question {i}", "theme": "atm"}) for i in range(4)]

    def iterator(**kwargs):
        try:
            yield from rows
        finally:
            state["closed"] += 1
    collection.iterator.side_effect = iterator
    collection.query.near_vector.return_value = NS(objects=rows[:2])
    connect = MagicMock(side_effect=AssertionError("must reuse supplied client"))
    monkeypatch.setattr("weaviate.connect_to_custom", connect)
    return client, collection, state, rows


@pytest.mark.parametrize("field", ["body", "default"])
def test_weaviate_injected_client_scans_searches_and_stays_open(monkeypatch, field):
    client, collection, state, rows = weaviate_client(monkeypatch, field=field)
    data = vt.from_client(client, "Questions", text_field="content", limits=vt.Limits(batch_size=2)).materialize()
    assert len(data.records) == 4 and all(r.text.startswith("ATM") for r in data.records)
    assert data.source["db_type"] == "weaviate" and data.source["vector_field"] == field
    assert state["closed"] == 1
    collection.iterator.assert_called_once_with(include_vector=True, cache_size=2)
    result = vt.from_client(client, "Questions", text_field="content", query_vector=[1, 2], search_limit=2).materialize()
    assert len(result.records) == 2 and result.scope == "retrieval_results"
    collection.query.near_vector.assert_called_once_with(near_vector=[1., 2.], limit=2, target_vector=field, include_vector=True)
    client.close.assert_not_called()


def test_cleanup_on_limit_failure_and_client_reuse(monkeypatch):
    client, collection, state, rows = weaviate_client(monkeypatch)
    with pytest.raises(vt.VTError, match="max_rows"):
        vt.from_client(client, "Questions", limits=vt.Limits(max_rows=1, batch_size=1)).materialize()
    assert state["closed"] == 1
    client.close.assert_not_called()
    assert len(vt.from_client(client, "Questions").materialize().records) == 4
    assert state["closed"] == 2
    client.close.assert_not_called()


@pytest.mark.parametrize("failure", ["duplicate", "dimension", "model", "revision", "bytes", "sdk_error"])
def test_invalid_database_reads_fail_without_closing_client(monkeypatch, failure):
    client, collection, state, rows = weaviate_client(monkeypatch)
    if failure == "duplicate":
        rows[1].uuid = rows[0].uuid
    elif failure == "dimension":
        rows[1].vector["body"] = [1.]
    elif failure in {"model", "revision"}:
        for i, row in enumerate(rows):
            row.properties[f"embedding_{failure}"] = f"different-{i}"
    elif failure == "bytes":
        rows[0].properties["content"] = "x" * 1500
    else:
        collection.iterator.side_effect = RuntimeError("secret-token https://private.example")
    with pytest.raises(vt.VTError) as error:
        vt.from_client(client, "Questions", limits=vt.Limits(max_bytes=1024, batch_size=1)).materialize()
    assert "secret-token" not in str(error.value) and "private.example" not in str(error.value)
    assert state["closed"] == (0 if failure == "sdk_error" else 1)
    client.close.assert_not_called()


def test_adapters_do_not_close_borrowed_clients(client, monkeypatch):
    close = MagicMock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    QdrantAdapter({"options": {}}, client=client).close()
    close.assert_not_called()
    weaviate, _, _, _ = weaviate_client(monkeypatch)
    WeaviateAdapter({"options": {}}, client=weaviate).close()
    weaviate.close.assert_not_called()


def test_empty_collection_and_portable_snapshot(client):
    client.create_collection("empty", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    data = vt.from_client(client, "empty").materialize()
    assert data.records == [] and data.scope == "complete_selection"
    assert json.loads(json.dumps(data.serializable()))["source"]["collection"] == "empty"
