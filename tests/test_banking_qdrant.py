from contextlib import closing
import numpy as np
import pytest
from qdrant_client import QdrantClient
from examples.banking77_qdrant import index_records, retrieve, query_and_plot
from test_density import fixture_data


def manifest():
    return {'sha256':'test-snapshot', 'dimensions':3, 'model':'test-fixture', 'model_revision':'fixed', 'rows':33}


def test_idempotent_index_query_and_snapshot_protection():
    with closing(QdrantClient(':memory:')) as client:
        data = fixture_data()
        metadata = manifest()
        assert index_records(client, 'banking', data.records, metadata) == 33
        assert index_records(client, 'banking', data.records, metadata) == 33
        records, hits = retrieve(client, 'banking', [1,0,0], metadata, 10)
        assert len(records)==10
        assert all(r.metadata['intent']=='card' for r in records)
        assert hits[0]['score'] >= hits[-1]['score']
        assert all(len(r.vector)==3 for r in records)
        with pytest.raises(ValueError, match='another dataset'):
            index_records(client, 'banking', data.records, {**metadata,'sha256':'different'})
        with pytest.raises(ValueError, match='fewer than three'):
            retrieve(client, 'banking', [1,0,0], {**metadata,'sha256':'missing'}, 10)


def test_query_plot_exports_actual_retrieved_records(tmp_path):
    pytest.importorskip('plotly')
    class Encoder:
        def encode(self, text, **kwargs):
            return np.array([1.,0.,0.])
    data=fixture_data()
    with closing(QdrantClient(':memory:')) as client:
        index_records(client, 'banking', data.records, manifest())
        report=query_and_plot(client, 'banking', manifest(), 'general banking', 33, tmp_path, data.limits, encoder=Encoder())
    assert report['retrieved']==33
    assert sum(g['queries'] for g in report['groups'])==33
    assert (tmp_path/'banking77_dbscan_3d.html').stat().st_size > 10000
    assert (tmp_path/'query_results.json').exists()
    assert (tmp_path/'query_report.md').exists()
