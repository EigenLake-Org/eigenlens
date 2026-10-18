import json
import numpy as np
import pytest
import eigenlens as vt
from eigenlens.registry import plan
from eigenlens.schemas import Limits, VTError


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
    with pytest.raises(ValueError,match="n_clusters"):
        plan([operation(n_clusters=2)])
    with pytest.raises(VTError,match="Unsupported"):
        plan([{"id":"c","type":"cluster","parameters":{"top_n":3}}])
    with pytest.raises(VTError,match="memory"):
        vt.analyze(fixture_data(limits=Limits(max_working_bytes=10000)),[operation()])


def test_small_silhouette_and_empty_vocabulary():
    data=fixture_data()
    for r in data.records:
        r.text="the a"
    result=vt.analyze(data,[operation(silhouette_sample_size=3)])["groups"]
    assert result.summary_data["vocabulary_status"]=="empty_vocabulary"
    json.dumps(result.model_dump(),allow_nan=False)


@pytest.mark.parametrize('components',[1,2,3,8])
def test_cached_search_matches_sklearn_euclidean_reference(components):
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(31)
    vectors=np.vstack([rng.normal(np.eye(8)[i],.025,(12,8)) for i in range(3)])
    # Include exact duplicates, whose off-diagonal distances must remain zero.
    vectors=np.vstack([vectors,vectors[:2]])
    data=vt.from_records([{'id':f'{i:03}','vector':v.tolist()} for i,v in enumerate(vectors)]).materialize()
    params={'pca_components':components,'eps_values':[.05,.15,.4,.9],
            'min_samples_values':[2,5],'silhouette_sample_size':20}
    result=vt.cluster(data,**params)
    assert result.effective_parameters['distance_cache']=='silhouette_subset'
    assert result.effective_parameters['neighborhood_cache']=='radius_graph'
    with threadpool_limits(limits=1):
        x=normalize(data.vectors())
        reduced=normalize(PCA(n_components=components,svd_solver='full',random_state=42).fit_transform(x))
        reference_best=None;reference_labels=None
        for row in result.summary_data['parameter_search']:
            labels=DBSCAN(eps=row['eps'],min_samples=row['min_samples'],n_jobs=1).fit_predict(reduced)
            idx=np.flatnonzero(labels!=-1)
            assert row['clusters']==len(set(labels)-{-1})
            assert row['noise_percentage']==pytest.approx(100*(1-len(idx)/len(labels)))
            if len(idx)>20:idx=np.sort(np.random.default_rng(42).choice(idx,20,replace=False))
            score=silhouette_score(reduced[idx],labels[idx]) if 2<=len(set(labels[idx]))<len(idx) else None
            assert row['silhouette']==pytest.approx(score,abs=1e-12) if score is not None else row['silhouette'] is None
            if score is not None and row['clusters']>=2 and row['noise_percentage']<70 and (reference_best is None or score>reference_best):
                reference_best=score;reference_labels=labels
        expected=reference_labels if reference_labels is not None else np.full(len(x),-1)
        np.testing.assert_array_equal([r['cluster_id'] for r in result.rows],expected)
        np.testing.assert_allclose([r['pca_3d'] for r in result.rows],PCA(n_components=3,svd_solver='full').fit_transform(x),atol=1e-12)


def test_distance_cache_falls_back_without_dropping_records():
    data=fixture_data(limits=Limits(max_working_bytes=45000))
    fallback=vt.analyze(data,[operation()])['groups']
    cached=vt.analyze(fixture_data(),[operation()])['groups']
    assert fallback.effective_parameters['distance_cache']=='disabled_memory_budget'
    assert fallback.processed==cached.processed==33
    assert fallback.rows==cached.rows


def test_numeric_budget_is_scoped_to_one_execution():
    from eigenlens.execution_context import numeric_budget,numeric_threads
    assert numeric_threads()==1
    with numeric_budget(2):
        assert vt.cluster(fixture_data()).effective_parameters['numeric_threads']==2
    assert numeric_threads()==1


def test_neighbor_cache_falls_back_within_memory_budget():
    # Dense distances fit, but retaining graph edges and DBSCAN copies does not.
    data=fixture_data(limits=Limits(max_working_bytes=60000))
    result=vt.analyze(data,[operation()])['groups']
    assert result.effective_parameters['distance_cache']=='silhouette_subset'
    assert result.effective_parameters['neighborhood_cache']=='disabled_memory_budget'
    assert result.rows==vt.analyze(fixture_data(),[operation()])['groups'].rows
