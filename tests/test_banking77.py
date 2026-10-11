"""The optional banking demo indexes records without changing the core API."""
from contextlib import closing
import hashlib
import json
from types import SimpleNamespace
import pytest
pytest.importorskip('qdrant_client')
from qdrant_client import QdrantClient
from examples.banking77 import index_records, read_records
from test_density import fixture_data


def manifest():
    return {'sha256':'test-snapshot', 'dimensions':3, 'model':'test-fixture',
            'model_revision':'fixed', 'rows':33, 'selection':'train_plus_test'}


def test_index_is_idempotent_and_preserves_embedding_provenance():
    with closing(QdrantClient(':memory:')) as client:
        data = fixture_data()
        meta = manifest()
        assert index_records(client, 'banking', data.records, meta) == 33
        assert index_records(client, 'banking', data.records, meta) == 33
        points = client.query_points('banking', using='body', query=[1,0,0], limit=10,
                                     with_payload=True, with_vectors=True).points
        assert len(points) == 10
        assert all(p.payload['intent']=='card' for p in points)
        assert all(p.payload['embedding_model']==meta['model'] and
                   p.payload['embedding_revision']==meta['model_revision'] for p in points)
        assert all(len(p.vector['body'])==3 for p in points)
        with pytest.raises(ValueError, match='another dataset'):
            index_records(client, 'banking', data.records, {**meta,'sha256':'different'})
        assert client.count('banking', exact=True).count == 33


def test_cache_requires_matching_checksum(tmp_path):
    path = tmp_path/'ingress/local/banking77.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(r.model_dump_json() for r in fixture_data().records)+'\n')
    meta = {**manifest(), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix('.manifest.json').write_text(json.dumps(meta))
    config = SimpleNamespace(root=tmp_path)
    records, saved = read_records(config)
    assert len(records)==33 and saved==meta
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='checksum'):
        read_records(config)
