"""Deterministic tests for temporal helpers and first-class temporal operators (no DB)."""
from datetime import datetime, timezone

import numpy as np
import pytest

import vector_tools as vt
from vector_tools.registry import plan
from vector_tools.temporal import (
    bucket_key,
    centroid,
    cohesion,
    cosine_drift,
    jensen_shannon,
    parse_time,
)


def _time_records():
    """Two tight directional groups in Jan, one shifted group in Feb, plus undated."""
    rng = np.random.default_rng(7)
    recs = []
    for i in range(12):
        v = np.array([5.0, 0.5]) + rng.normal(0, 0.05, 2)
        recs.append({"id": f"jan-{i:02d}", "vector": v.tolist(), "text": f"jan story {i}",
                     "metadata": {"published_at": f"2023-01-{(i % 28) + 1:02d}"}})
    for i in range(12):
        v = np.array([0.5, 5.0]) + rng.normal(0, 0.05, 2)
        recs.append({"id": f"feb-{i:02d}", "vector": v.tolist(), "text": f"feb story {i}",
                     "metadata": {"published_at": f"2023-02-{(i % 28) + 1:02d}"}})
    recs.append({"id": "nodate", "vector": [5.0, 0.5], "text": "no time",
                 "metadata": {}})
    return recs


def test_parse_published_at_variants():
    assert parse_time("2023-03-23 07:40:00") == datetime(2023, 3, 23, 7, 40, tzinfo=timezone.utc)
    assert parse_time("2023-08-04") == datetime(2023, 8, 4, tzinfo=timezone.utc)
    assert parse_time("2023-08-04T12:00:00") == datetime(2023, 8, 4, 12, 0, tzinfo=timezone.utc)
    assert parse_time(None) is None
    assert parse_time("") is None
    assert parse_time("not-a-date") is None
    assert bucket_key(datetime(2023, 8, 4, tzinfo=timezone.utc), "month") == "2023-08"
    assert bucket_key(datetime(2023, 8, 4, tzinfo=timezone.utc), "week").startswith("2023-W")


def test_library_parse_time_accepts_objects():
    assert parse_time(datetime(2023, 1, 2, tzinfo=timezone.utc)) == datetime(2023, 1, 2, tzinfo=timezone.utc)
    assert parse_time(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert bucket_key(datetime(2023, 1, 15, tzinfo=timezone.utc), "day") == "2023-01-15"
    assert bucket_key(datetime(2023, 1, 15, tzinfo=timezone.utc), "month") == "2023-01"
    with pytest.raises(ValueError, match="Unknown bucket"):
        bucket_key(datetime(2023, 1, 15, tzinfo=timezone.utc), "year")


def test_centroid_drift_cohesion():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = centroid(a)
    assert c.shape == (2,)
    assert abs(float(np.linalg.norm(c)) - 1.0) < 1e-9
    assert cosine_drift(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert abs(cosine_drift(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-9
    assert 0.0 < cohesion(a, c) <= 1.0


def test_jensen_shannon_bounds():
    p = np.array([0.5, 0.5])
    assert jensen_shannon(p, p) == 0.0
    jsd = jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert abs(jsd - 1.0) < 1e-9


def test_drift_detects_shift():
    data = vt.from_records(_time_records()).materialize()
    res = vt.drift(data, time_field="published_at", bucket="month", min_bucket_n=5)
    assert [r["bucket"] for r in res.rows] == ["2023-01", "2023-02"]
    assert res.rows[0]["drift_prev"] is None
    assert res.rows[1]["drift_prev"] > 0.5  # near-orthogonal Jan vs Feb clouds
    assert res.rows[1]["drift_first"] == res.rows[1]["drift_prev"]
    assert res.processed == 24
    assert res.skipped["missing_time"] == 1
    assert res.summary_data["bucket_count"] == 2
    assert len(res.examples()) == 2


def test_drift_stable_when_no_shift():
    recs = [{"id": f"a-{i}", "vector": [5.0, 0.1 * i], "metadata": {"published_at": "2023-01-05"}}
            for i in range(6)]
    recs += [{"id": f"b-{i}", "vector": [5.0, 0.1 * i], "metadata": {"published_at": "2023-02-05"}}
             for i in range(6)]
    res = vt.drift(vt.from_records(recs).materialize(), time_field="published_at", min_bucket_n=5)
    assert res.rows[1]["drift_prev"] < 0.01


def test_drift_validation():
    data = vt.from_records(_time_records()).materialize()
    with pytest.raises(vt.VTError, match="at least two time buckets"):
        vt.drift(data, time_field="published_at", min_bucket_n=50)
    with pytest.raises(vt.VTError, match="No parsable time"):
        vt.drift(data, time_field="absent_field")
    with pytest.raises(vt.VTError, match="bucket must be"):
        vt.drift(data, time_field="published_at", bucket="year")


def test_novelty_ranks_late_outlier_first():
    recs = [{"id": f"base-{i}", "vector": [5.0, 0.2], "metadata": {"published_at": "2023-01-10"}}
            for i in range(8)]
    recs += [{"id": "late-typical", "vector": [5.0, 0.25], "metadata": {"published_at": "2023-02-10"}}]
    recs += [{"id": "late-outlier", "vector": [0.2, 5.0], "metadata": {"published_at": "2023-02-11"}}]
    data = vt.from_records(recs).materialize()
    res = vt.novelty(data, time_field="published_at", min_bucket_n=2)
    assert res.rows[0]["id"] == "late-outlier"
    assert res.rows[0]["novelty"] > res.rows[1]["novelty"]
    assert res.summary_data["probe_records"] == 2
    assert res.skipped["no_history"] == 8


def test_flow_tracks_cluster_shift():
    recs = []
    for i in range(10):  # Jan dominated by direction A
        recs.append({"id": f"jan-a-{i}", "vector": [5.0, 0.1], "metadata": {"published_at": "2023-01-05"}})
    for i in range(2):
        recs.append({"id": f"jan-b-{i}", "vector": [0.1, 5.0], "metadata": {"published_at": "2023-01-06"}})
    for i in range(2):  # Feb flips to direction B
        recs.append({"id": f"feb-a-{i}", "vector": [5.0, 0.1], "metadata": {"published_at": "2023-02-05"}})
    for i in range(10):
        recs.append({"id": f"feb-b-{i}", "vector": [0.1, 5.0], "metadata": {"published_at": "2023-02-06"}})
    data = vt.from_records(recs).materialize()
    results = vt.analyze(data, [
        {"id": "groups", "type": "cluster", "parameters": {"n_clusters": 2}},
        {"id": "fl", "type": "flow", "clusters_from": "groups",
         "parameters": {"time_field": "published_at", "bucket": "month", "min_bucket_n": 5}},
    ])
    flow = results["fl"]
    assert [r["bucket"] for r in flow.rows] == ["2023-01", "2023-02"]
    assert flow.rows[1]["jsd_vs_prev"] > 0.3
    assert flow.rows[0]["largest_share"] > 0.7


def test_registry_temporal_validation():
    ordered = plan([
        {"id": "d", "type": "drift", "parameters": {"time_field": "published_at", "bucket": "week"}},
        {"id": "n", "type": "novelty"},
    ], "local")
    assert {o.id for o in ordered} == {"d", "n"}
    with pytest.raises(vt.VTError, match="Flow requires clusters_from"):
        plan([{"id": "f", "type": "flow"}], "local")
    with pytest.raises(vt.VTError, match="Unsupported parameters"):
        plan([{"id": "d", "type": "drift", "parameters": {"n_clusters": 3}}], "local")
    with pytest.raises(vt.VTError, match="Only topics, sentiment and flow accept"):
        plan([{"id": "c", "type": "cluster"}, {"id": "d", "type": "drift", "clusters_from": "c"}], "local")
