"""SDK-mocked contract checks; these are explicitly not live remote integrations."""
from types import SimpleNamespace as NS
import pytest
import numpy as np
import vector_tools as vt
from vector_tools.schemas import Selection
from vector_tools.sources import Pinecone, Weaviate, Qdrant, Pgvector
from vector_tools.demo import tickets
from examples.seed_sources import fixture, payload

class FakeIndex:
    config = NS(host="test-index")
    def __init__(self):
        self.calls = []
        self.data = {r.id:r for r in fixture()}
    def list(self, namespace, limit):
        self.calls.append(("list",namespace))
        ids = sorted(self.data)
        for i in range(0,len(ids),limit):
            yield ids[i:i+limit]
    def fetch(self, ids, namespace):
        self.calls.append(("fetch",namespace))
        from pinecone import FetchResponse, Vector
        return FetchResponse(namespace=namespace, vectors={id:Vector(id=id, values=self.data[id].vector, metadata=payload(self.data[id])) for id in reversed(ids) if id in self.data})

def test_pinecone_mock_pages_scope_capability():
    index = FakeIndex()
    source = Pinecone(index, namespace="scope-a", index_description={"spec":{"serverless":{"cloud":"aws"}}, "vector_type":"dense", "dimension":4})
    data = vt.Dataset(source, limits=vt.Limits(batch_size=7)).materialize()
    assert len(data.records) == 62
    assert all(scope == "scope-a" for _,scope in index.calls)
    for r in data.records:
        assert r.text == index.data[r.id].text and r.vector == index.data[r.id].vector
    other = vt.Dataset(Pinecone(index, namespace="scope-b", index_description={"spec":{"serverless":{"cloud":"aws"}}})).materialize()
    assert data.fingerprint != other.fingerprint
    unsupported = vt.Dataset(Pinecone(index, namespace="scope-a"))
    with pytest.raises(vt.VTError, match="enumerate"):
        unsupported.sample(5).materialize()
    assert len(unsupported.ids([data.records[0].id]).materialize().records) == 1
    with pytest.raises(vt.VTError, match="sparse"):
        Pinecone(index, namespace="x", index_description={"vector_type":"sparse"})

class FakeWeaviateCollection:
    def __init__(self, rows, tenant=None):
        self.rows, self.tenant = rows, tenant
        self.config = NS(get=lambda:NS(multi_tenancy_config=NS(enabled=True), vector_config={"body":{}}, vector_index_config=None))
        self.query = NS(fetch_object_by_id=self.fetch)
    def with_tenant(self, tenant):
        return FakeWeaviateCollection(self.rows,tenant)
    def fetch(self, id, include_vector):
        assert self.tenant == "a" and include_vector
        r = next((r for r in self.rows if r.id == str(id)),None)
        return NS(uuid=r.id, vector={"body":r.vector}, properties=payload(r)) if r else None
    def iterator(self, include_vector, cache_size):
        assert cache_size == 7
        for r in sorted(self.rows,key=lambda r:r.id):
            yield self.fetch(r.id, include_vector)

def test_weaviate_mock_tenant_named_vector():
    client = NS(collections=NS(get=lambda name:FakeWeaviateCollection(fixture())))
    ds = vt.from_weaviate(client, collection="Tickets",tenant="a",vector_field="body")
    data = vt.Dataset(ds.source, limits=vt.Limits(batch_size=7)).materialize()
    assert len(data.records) == 62 and data.source["tenant"] == "a"
    assert len(ds.ids([data.records[0].id]).materialize().records) == 1
    with pytest.raises(vt.VTError, match="tenant"):
        vt.from_weaviate(client, collection="Tickets",vector_field="body").materialize()

def test_postgres_sql_quote_and_parameterize():
    source = Pgvector(None, schema='odd"schema', table='table; DROP TABLE tickets', metadata_columns=["theme"])
    attack = "x' OR 1=1 --"
    query, args, cols = source.query(Selection(filters=[vt.Predicate(field="theme",value=attack)]), {"vector"})
    sql = query.as_string()
    assert '"odd""schema"' in sql and '"table; DROP TABLE tickets"' in sql
    assert attack not in sql and args == [attack] and "%s" in sql
    with pytest.raises(vt.VTError, match="not configured"):
        source.query(Selection(filters=[vt.Predicate(field="secret",value=1)]), {"vector"})

def test_qdrant_real_local_client():
    # Qdrant client's actual in-process database, not a mock or remote Docker server.
    from qdrant_client import QdrantClient, models
    client=QdrantClient(":memory:")
    client.create_collection("tickets", vectors_config={"body":models.VectorParams(size=4,distance=models.Distance.DOT)})
    client.upsert("tickets", [models.PointStruct(id=r.id, vector={"body":r.vector},payload=payload(r)) for r in fixture()])
    source=Qdrant(client,"tickets",vector_field="body")
    ds=vt.Dataset(source,limits=vt.Limits(batch_size=7))
    data=ds.materialize()
    assert len(data.records)==62
    expected={r.id:r for r in fixture()}
    for r in data.records:
        assert r.text==expected[r.id].text
        assert np.allclose(r.vector,expected[r.id].vector)
    assert len(ds.filter(theme="login").materialize().records)==20
    assert len(ds.ids([r.id for r in data.records[:3]]).materialize().records)==3
    client.close()
