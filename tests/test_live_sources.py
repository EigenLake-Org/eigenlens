"""Live checks only run when VECTOR_ML_LIVE=1. Seed test scopes first."""
import os
import numpy as np
import pytest
import vector_tools as vt
from vector_tools.config import Config, close_source
from examples.seed_sources import fixture
from examples.mcp_client import OPERATIONS

@pytest.mark.live
@pytest.mark.parametrize("source_id",["qdrant","weaviate","pgvector","pinecone"])
def test_live_source_portability(source_id):
    if os.getenv("VECTOR_ML_LIVE") != "1":
        pytest.skip("Set VECTOR_ML_LIVE=1 and seed configured demo scopes")
    if source_id == "pinecone" and not os.getenv("PINECONE_API_KEY"):
        pytest.skip("Pinecone credentials absent: live integration unverified")
    source = Config("config/local.json").source(source_id)
    try:
        ds=vt.Dataset(source,limits=vt.Limits(batch_size=7))
        data=ds.materialize()
        expected={r.id:r for r in fixture()}
        assert {r.id for r in data.records}==set(expected)
        for r in data.records:
            assert r.text==expected[r.id].text
            # Qdrant cosine storage normalizes vectors. Compare direction, not magnitude.
            a,b=np.asarray(r.vector),np.asarray(expected[r.id].vector)
            assert np.allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=1e-5)
        retrieved=ds.ids(list(expected)[:9]).materialize()
        assert len(retrieved.records)==9 and retrieved.scope=="retrieval_results"
        assert len(ds.filter(theme="billing").materialize().records)==20
        a,b=ds.sample(11,42).materialize(),ds.sample(11,42).materialize()
        assert a.fingerprint==b.fingerprint and len(a.records)==11
        results=vt.analyze(data,OPERATIONS)
        assert all(r.dataset_fingerprint==data.fingerprint for r in results.values())
        assert len(results["groups"].rows)==62
        assert results["sentiment"].summary_data["labels"]=={"negative":42,"positive":20}
    finally:
        close_source(source)
