import importlib.util
import numpy as np
import pytest
import vector_tools as vt
from vector_tools.demo import tickets
from vector_tools.registry import plan
from vector_tools.schemas import Selection
from vector_tools.sources.base import InMemory

@pytest.fixture
def data():
    return vt.from_records(tickets()).materialize()

def test_real_analysis(data):
    results = vt.analyze(data, [
        {"id": "themes", "type": "topics", "clusters_from": "groups"},
        {"id": "groups", "type": "cluster", "parameters": {"n_clusters": 4}},
        {"id": "sent", "type": "sentiment", "clusters_from": "groups"},
        {"id": "odd", "type": "outliers"}])
    assert sum(results["groups"].summary_data["cluster_sizes"].values()) == 62
    assert {r["id"] for r in results["groups"].rows} == {r.id for r in data.records}
    terms = {t["term"] for r in results["themes"].rows for t in r["terms"]}
    assert {"login", "billing", "sync"} <= terms
    assert {r["id"] for r in results["odd"].top(2)} == {"unusual-0", "unusual-1"}
    assert all(r["anomaly_score"] == -r["raw_score"] for r in results["odd"].rows)
    assert results["sent"].summary_data["labels"]["negative"] >= 40
    assert results["sent"].summary_data["labels"]["positive"] == 20
    assert sum(g["denominator"] for g in results["sent"].summary_data["by_cluster"].values()) == 62
    assert data.to_arrow().num_rows == 62

def test_sampling_and_scope():
    a = vt.from_records(tickets()).sample(10, 42).materialize()
    b = vt.from_records(list(reversed(tickets()))).sample(10, 42).materialize()
    assert a.fingerprint == b.fingerprint
    assert a.scope == "sample" and len(a.records) == 10 and a.read_count == 62
    retrieved = vt.from_records(tickets()).ids(["billing-000", "missing"]).materialize()
    assert retrieved.scope == "retrieval_results" and retrieved.missing_ids == ["missing"]
    assert vt.from_records(tickets()).filter(theme="login").materialize().scope == "filtered_selection"

def test_invalid_and_limits(data):
    with pytest.raises(ValueError, match="finite"):
        vt.Record(id="bad", vector=[float("nan")])
    with pytest.raises(vt.VTError, match="dimension"):
        vt.from_records([{"id":"a", "vector":[1]}, {"id":"b", "vector":[1,2]}]).materialize()
    with pytest.raises(vt.VTError, match="max_rows"):
        vt.from_records(tickets(), limits=vt.Limits(max_rows=3)).materialize()
    with pytest.raises(vt.VTError, match="max_scan_rows"):
        vt.from_records(tickets(), limits=vt.Limits(max_scan_rows=3)).sample(2).materialize()
    with pytest.raises(vt.VTError, match="n_clusters"):
        vt.cluster(data, n_clusters=100)
    with pytest.raises(vt.VTError, match="Empty"):
        vt.cluster(vt.from_records([]).materialize())
    c = vt.cluster(data, n_clusters=3)
    with pytest.raises(vt.VTError, match="same materialized"):
        vt.topics(vt.from_records(tickets()).sample(5).materialize(), c)

def test_empty_text():
    data = vt.from_records([{"id":"a", "vector":[1,2]}, {"id":"b", "vector":[2,3], "text":"the and"}]).materialize()
    c = vt.cluster(data, n_clusters=1)
    t = vt.topics(data, c)
    assert t.summary_data["vocabulary_status"] == "empty_vocabulary"
    assert t.skipped["missing_text"] == 1
    assert vt.sentiment(data).processed == 1

def test_topic_phrases_do_not_cross_record_boundaries():
    data = vt.from_records([
        {"id":"a", "vector":[1,0], "text":"cash withdrawal"},
        {"id":"b", "vector":[1,.01], "text":"card payment"},
    ]).materialize()
    results = vt.analyze(data, [
        {"id":"groups", "type":"cluster", "parameters":{"n_clusters":1}},
        {"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":2}},
    ])
    terms={t['term'] for row in results['themes'].rows for t in row['terms']}
    assert {'cash withdrawal','card payment'} <= terms
    assert 'withdrawal card' not in terms
    assert results['themes'].effective_parameters['ngram_range']==[1,2]
    assert results['themes'].processed==2
    with pytest.raises(ValueError):
        plan([{"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":4}}])

def test_registry_rejection():
    for ops, backend in [([{"id":"g", "type":"cluster"}], "spark"),
                          ([{"id":"t", "type":"topics", "clusters_from":"absent"}], "local"),
                          ([{"id":"g", "type":"cluster", "parameters":{"n_estimators":4}}], "local"),
                          ([{"id":"g", "type":"cluster"}]*2, "local")]:
        with pytest.raises(vt.VTError):
            plan(ops, backend)

def test_filter_null_semantics():
    rows = [{"id":"1", "metadata":{"x":1}}, {"id":"2", "metadata":{"x":None}}, {"id":"3"}]
    assert [r.id for r in vt.from_records(rows).filter({"field":"x", "op":"ne", "value":2}).materialize().records] == ["1"]
    assert len(vt.from_records(rows).filter({"field":"x", "op":"exists", "value":False}).materialize().records) == 2

def test_model_availability_error(data, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "vaderSentiment.vaderSentiment", None)
    with pytest.raises(vt.VTError, match="weights unavailable"):
        vt.sentiment(data)

def test_scope_identity_and_content_copy():
    r=vt.Record(id="same",vector=[1,2],text="original")
    ds=vt.from_records([r],name="first")
    r.text="changed outside dataset"
    a=ds.materialize()
    b=vt.from_records([{"id":"same","vector":[1,2],"text":"original"}],name="second").materialize()
    assert a.records[0].text=="original"
    assert a.fingerprint!=b.fingerprint

def test_metadata_dates_serialize_portably():
    from datetime import datetime, timezone
    from vector_tools.datasets import canonical
    data=vt.from_records([{"id":"dated", "vector":[1,2], "metadata":{"created":datetime(2026,1,1,tzinfo=timezone.utc)}}]).materialize()
    assert "2026-01-01" in canonical(data.serializable())
    assert data.to_arrow().num_rows==1

def test_direct_estimator_objects(data):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import IsolationForest
    clusters=vt.cluster(data,estimator=MiniBatchKMeans(n_clusters=4,random_state=73,n_init=3))
    odd=vt.outliers(data,estimator=IsolationForest(random_state=73,n_estimators=50))
    assert clusters.effective_parameters["random_state"]==73
    assert odd.effective_parameters["n_estimators"]==50
    assert odd.effective_parameters["random_state"]==73
    assert {r["id"] for r in odd.top(2)}=={"unusual-0","unusual-1"}
