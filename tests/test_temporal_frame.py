"""Tests for TemporalFrame, frame sharing, registry metadata, and evidence packs (no DB)."""
import numpy as np
import pytest

import vector_tools as vt
from vector_tools import TemporalFrame, clear_frame_cache, get_frame
from vector_tools.evidence import render_evidence, render_result
from vector_tools.registry import REGISTRY
from vector_tools.schemas import VTError


def _dated(n_per=6):
    recs = []
    for i in range(n_per):
        recs.append({"id": f"a-{i}", "vector": [5.0, 0.5], "text": f"alpha story {i}",
                     "metadata": {"published_at": "2023-01-10"}})
    for i in range(n_per):
        recs.append({"id": f"b-{i}", "vector": [0.5, 5.0], "text": f"beta story {i}",
                     "metadata": {"published_at": "2023-02-10"}})
    return recs


def _materialized(recs):
    return vt.from_records(recs).materialize()


def test_frame_build_basics():
    data = _materialized(_dated())
    frame = TemporalFrame.build(data, time_field="published_at", bucket="month", min_bucket_n=5)
    assert frame.order == ("2023-01", "2023-02")
    assert frame.dropped == ()
    assert frame.missing_time == 0
    assert frame.kept_records == 12
    assert frame.matrix.shape == (12, 2)


def test_frame_invalid_params():
    data = _materialized(_dated())
    with pytest.raises(VTError):
        TemporalFrame.build(data, bucket="year")
    # Empty time_field follows the operator defaulting convention.
    assert TemporalFrame.build(data, time_field="").time_field == "published_at"
    with pytest.raises(VTError):
        TemporalFrame.build(data, min_bucket_n=1)
    with pytest.raises(VTError):
        TemporalFrame.build(data, time_field="nope")


def test_frame_stable_under_record_reorder():
    recs = _dated()
    f1 = TemporalFrame.build(_materialized(recs))
    f2 = TemporalFrame.build(_materialized(list(reversed(recs))))
    assert f1.order == f2.order == ("2023-01", "2023-02")
    assert {k: len(v) for k, v in f1.buckets.items()} == {k: len(v) for k, v in f2.buckets.items()}


def test_frame_cache_shares_one_object():
    clear_frame_cache()
    data = _materialized(_dated())
    f1 = get_frame(data, bucket="month")
    f2 = get_frame(data, bucket="month")
    assert f1 is f2
    f3 = get_frame(data, bucket="week")
    assert f3 is not f1
    clear_frame_cache()
    assert get_frame(data, bucket="month") is not f1


def test_execute_shares_frame_between_temporal_ops():
    clear_frame_cache()
    data = _materialized(_dated())
    res = vt.analyze(data, [
        {"id": "d", "type": "drift", "parameters": {"time_field": "published_at", "bucket": "month"}},
        {"id": "n", "type": "novelty", "parameters": {"time_field": "published_at", "bucket": "month"}},
    ])
    assert set(res) == {"d", "n"}
    # The backend path populates the same content-addressed cache.
    assert get_frame(data, "published_at", "month", 5, True).order == ("2023-01", "2023-02")


def test_drift_identical_buckets_is_zero():
    recs = [{"id": f"a-{i}", "vector": [5.0, 0.1 * i], "metadata": {"published_at": "2023-01-05"}}
            for i in range(6)]
    recs += [{"id": f"b-{i}", "vector": [5.0, 0.1 * i], "metadata": {"published_at": "2023-02-05"}}
             for i in range(6)]
    res = vt.drift(_materialized(recs), min_bucket_n=5)
    assert res.rows[1]["drift_prev"] == 0.0


def test_flow_identical_distributions_zero_jsd():
    # Both months hold the same 3xA + 3xB mix, so histograms match exactly.
    recs = []
    for m, tag in (("2023-01-10", "a"), ("2023-02-10", "b")):
        for i in range(3):
            recs.append({"id": f"{tag}-A-{i}", "vector": [5.0, 0.5], "text": "aaa",
                         "metadata": {"published_at": m}})
            recs.append({"id": f"{tag}-B-{i}", "vector": [0.5, 5.0], "text": "bbb",
                         "metadata": {"published_at": m}})
    data = _materialized(recs)
    clusters = vt.cluster(data, n_clusters=2, random_state=42)
    res = vt.flow(data, clusters, time_field="published_at", bucket="month", min_bucket_n=5)
    assert res.rows[1]["jsd_vs_prev"] == 0.0


def test_registry_temporal_metadata():
    for op in ("drift", "novelty", "flow"):
        assert REGISTRY[op].get("temporal") is True
        assert {"time_field", "bucket", "min_bucket_n"} <= set(REGISTRY[op]["frame_params"])
    assert "temporal" not in REGISTRY["cluster"]
    assert "temporal" not in REGISTRY["topics"]


def test_evidence_happy_path():
    data = _materialized(_dated())
    clusters = vt.cluster(data, n_clusters=2, random_state=42)
    results = {
        "d": vt.drift(data, time_field="published_at", bucket="month", min_bucket_n=5),
        "n": vt.novelty(data, time_field="published_at", bucket="month", min_bucket_n=5),
        "g": clusters,
        "f": vt.flow(data, clusters, time_field="published_at", bucket="month", min_bucket_n=5),
        "t": vt.topics(data, clusters, top_n=4),
    }
    pack = render_evidence(results)
    assert "evidence/1.0" in pack
    assert "beta story" in pack  # Feb probes carry text; Jan is history-only
    assert "2023-01" in pack and "2023-02" in pack
    assert "terms=" in pack  # topics lookup joined into flow lines
    assert len(pack) <= 12000


def test_evidence_rejects_novelty_without_text():
    data = _materialized(_dated())
    res = vt.novelty(data, time_field="published_at", bucket="month", min_bucket_n=5)
    stripped = res.model_copy(update={"example_rows": [
        {**e, "text": ""} for e in res.example_rows]})
    with pytest.raises(VTError, match="record text"):
        render_result(stripped)


def test_evidence_rejects_zero_shares():
    data = _materialized(_dated())
    clusters = vt.cluster(data, n_clusters=2, random_state=42)
    res = vt.flow(data, clusters, time_field="published_at", bucket="month", min_bucket_n=5)
    zeroed = res.model_copy(update={"rows": [
        {**r, "largest_share": 0.0} for r in res.rows]})
    with pytest.raises(VTError, match="degenerate"):
        render_result(zeroed)


def test_evidence_rejects_single_bucket_and_mixed_datasets():
    data = _materialized(_dated())
    res = vt.drift(data, time_field="published_at", bucket="month", min_bucket_n=5)
    with pytest.raises(VTError):
        render_result(res.model_copy(update={"rows": res.rows[:1]}))
    other = vt.drift(_materialized(_dated() + [
        {"id": "extra", "vector": [1.0, 1.0], "text": "x", "metadata": {"published_at": "2023-03-01"}}]),
        time_field="published_at", bucket="month", min_bucket_n=5)
    with pytest.raises(VTError, match="one dataset"):
        render_evidence({"a": res, "b": other})


def test_evidence_truncates():
    data = _materialized(_dated())
    res = vt.drift(data, time_field="published_at", bucket="month", min_bucket_n=5)
    pack = render_evidence({"d": res}, max_chars=50)
    assert pack.endswith("[truncated]")
