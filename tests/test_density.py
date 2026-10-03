import json
import numpy as np
import pytest
import vector_tools as vt
from vector_tools.registry import plan
from vector_tools.schemas import Limits, VTError


def fixture_data(labels=True, limits=None):
    rng = np.random.default_rng(17)
    x = np.vstack([rng.normal([1,0,0], .012, (16,3)), rng.normal([0,1,0], .012, (16,3)), [[0,0,1]]])
    return vt.from_records([vt.Record(id=str(i), vector=v.tolist(), text="card payment" if i<16 else "cash withdrawal",
                           metadata={"intent": "card" if i<16 else "cash"} if labels else {}) for i,v in enumerate(x)], limits=limits).materialize()


def operation(**params):
    return {"id":"groups", "type":"cluster", "algorithm":"dbscan_search", "parameters":{
        "pca_components":3, "eps_values":[.1,.2], "min_samples_values":[4], "label_field":"intent", **params}}


def test_search_noise_evaluation_and_topic_dependency():
    data=fixture_data()
    results=vt.analyze(data,[operation(), {"id":"topics","type":"topics","clusters_from":"groups"}])
    result=results["groups"]
    assert result.summary_data["status"] == "selected"
    assert result.summary_data["noise_count"] == 1
    assert sorted(result.summary_data["cluster_sizes"].values()) == [1,16,16]
    assert len(result.rows)==33 and all(len(r["pca_3d"])==3 for r in result.rows)
    assert result.summary_data["evaluation"]["labels_used_for_selection"] is False
    assert result.summary_data["evaluation"]["adjusted_rand_index"] > .9
    assert results["topics"].processed==33
    json.dumps(result.model_dump(), allow_nan=False)
    without=vt.analyze(fixture_data(labels=False),[operation()])["groups"]
    assert without.rows==result.rows
    assert without.summary_data["evaluation"]["status"]=="unavailable"


def test_no_candidate_preserves_search():
    result=vt.analyze(fixture_data(),[operation(min_samples_values=[100])])["groups"]
    assert result.summary_data["status"]=="no_valid_candidate"
    assert len(result.summary_data["parameter_search"])==2
    assert result.effective_parameters["best_parameters"] is None
    assert all(r["cluster_id"]==-1 for r in result.rows)
    json.dumps(result.model_dump(),allow_nan=False)


@pytest.mark.parametrize("params",[{"eps_values":[0]}, {"eps_values":[float('nan')]}, {"min_samples_values":[0]}, {"eps_values":[]}])
def test_invalid_grid(params):
    with pytest.raises(ValueError):
        plan([operation(**params)])


def test_algorithm_parameter_isolation_and_memory():
    with pytest.raises(VTError,match="Unsupported"):
        plan([operation(n_clusters=2)])
    with pytest.raises(VTError,match="Unsupported"):
        plan([{"id":"c","type":"cluster","parameters":{"eps_values":[.3]}}])
    with pytest.raises(VTError,match="memory"):
        vt.analyze(fixture_data(limits=Limits(max_working_bytes=10000)),[operation()])


def test_small_silhouette_and_empty_vocabulary():
    data=fixture_data()
    for r in data.records:
        r.text="the a"
    result=vt.analyze(data,[operation(silhouette_sample_size=3)])["groups"]
    assert result.summary_data["vocabulary_status"]=="empty_vocabulary"
    json.dumps(result.model_dump(),allow_nan=False)
