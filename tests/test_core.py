import pytest
import eigenlens as vt
from eigenlens.demo import tickets
from eigenlens.registry import plan
from test_density import fixture_data

@pytest.fixture
def data():
    return vt.from_records(tickets()).materialize()

def test_real_analysis(data):
    results = vt.analyze(data, [
        {"id": "themes", "type": "topics", "clusters_from": "groups"},
        {"id": "groups", "type": "cluster", "algorithm":"dbscan_search"}])
    assert sum(results["groups"].summary_data["cluster_sizes"].values()) == 62
    assert {r["id"] for r in results["groups"].rows} == {r.id for r in data.records}
    terms = {t["term"] for r in results["themes"].rows for t in r["terms"]}
    assert {"login", "billing", "sync"} <= terms

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
    with pytest.raises(ValueError, match="n_clusters"):
        vt.cluster(data, n_clusters=100)
    with pytest.raises(vt.VTError, match="Empty"):
        vt.cluster(vt.from_records([]).materialize())
    c = vt.cluster(data)
    with pytest.raises(vt.VTError, match="same materialized"):
        vt.topics(vt.from_records(tickets()).sample(5).materialize(), c)

def test_empty_text():
    data = fixture_data()
    for i, record in enumerate(data.records):
        record.text = "the and" if i == 0 else None
    c = vt.cluster(data)
    t = vt.topics(data, c)
    assert t.summary_data["vocabulary_status"] == "empty_vocabulary"
    assert t.skipped["missing_text"] == 32
    assert t.processed == 1

def test_topic_phrases_do_not_cross_record_boundaries():
    data = fixture_data()
    for i, record in enumerate(data.records):
        record.text = ["cash withdrawal", "card payment"][i] if i < 2 else None
    results = vt.analyze(data, [
        {"id":"groups", "type":"cluster"},
        {"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":2}},
    ])
    terms={t['term'] for row in results['themes'].rows for t in row['terms']}
    assert {'cash withdrawal','card payment'} <= terms
    assert 'withdrawal card' not in terms
    assert not any('vectorrecordboundary' in term for term in terms)
    assert results['themes'].effective_parameters['ngram_range']==[1,2]
    assert results['themes'].processed==2
    assert results['themes'].versions['bertopic']
    with pytest.raises(ValueError):
        plan([{"id":"themes", "type":"topics", "clusters_from":"groups", "parameters":{"ngram_max":4}}])

def test_registry_rejection():
    for ops, backend in [([{"id":"g", "type":"cluster"}], "spark"),
                          ([{"id":"t", "type":"topics", "clusters_from":"absent"}], "local"),
                          ([{"id":"g", "type":"cluster", "parameters":{"top_n":4}}], "local"),
                          ([{"id":"g", "type":"cluster"}]*2, "local")]:
        with pytest.raises(vt.VTError):
            plan(ops, backend)

def test_filter_null_semantics():
    rows = [{"id":"1", "metadata":{"x":1}}, {"id":"2", "metadata":{"x":None}}, {"id":"3"}]
    assert [r.id for r in vt.from_records(rows).filter({"field":"x", "op":"ne", "value":2}).materialize().records] == ["1"]
    assert len(vt.from_records(rows).filter({"field":"x", "op":"exists", "value":False}).materialize().records) == 2

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
    from eigenlens.datasets import canonical
    data=vt.from_records([{"id":"dated", "vector":[1,2], "metadata":{"created":datetime(2026,1,1,tzinfo=timezone.utc)}}]).materialize()
    assert "2026-01-01" in canonical(data.serializable())

def test_default_cluster_is_dbscan(data):
    clusters=vt.cluster(data)
    assert clusters.effective_parameters["algorithm"] == "dbscan_search"
    assert len(clusters.rows)==62
    with pytest.raises(vt.VTError, match="dbscan_search"):
        vt.cluster(data, algorithm="minibatch_kmeans")
    with pytest.raises(vt.VTError, match="dbscan_search"):
        plan([{"id":"old", "type":"cluster", "algorithm":"minibatch_kmeans"}])

@pytest.mark.parametrize("operation", ["sentiment", "outliers"])
def test_removed_operations_are_rejected(operation):
    with pytest.raises(ValueError):
        plan([{"id":"removed", "type":operation}])


def test_bertopic_preserves_original_ids_after_internal_renumbering():
    data = fixture_data()
    groups = vt.cluster(data)
    for i, (record, assignment) in enumerate(zip(data.records, groups.rows)):
        cid = 8 if i < 10 else 42 if i < 32 else -1
        assignment['cluster_id'] = cid
        record.text = 'card payment' if cid == 8 else 'cash withdrawal' if cid == 42 else 'unusual export'
    themes = vt.topics(data, groups, ngram_max=2)
    terms = {row['cluster_id']:{term['term'] for term in row['terms']} for row in themes.rows}
    assert set(terms) == {-1, 8, 42}
    assert 'card payment' in terms[8]
    assert 'cash withdrawal' in terms[42]
    assert 'unusual export' in terms[-1]
    assert themes.effective_parameters['topic_id_map'] == {'8':1, '42':0, '-1':-1}
    assert themes.effective_parameters['representation'] == 'bertopic'
    assert themes.versions['bertopic'] == '0.17.4'


def test_bertopic_one_word_vocabulary():
    data = fixture_data()
    for record in data.records:
        record.text = "ATM"
    themes = vt.topics(data, vt.cluster(data))
    assert themes.processed == 33
    assert themes.summary_data['vocabulary_status'] == 'ok'
    assert all(row['terms'][0]['term'] == 'atm' for row in themes.rows)
