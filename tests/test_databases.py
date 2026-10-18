"""Persistence, isolation, complete retrieval, SDK mapping and real Qdrant jobs."""
from contextlib import closing
import json
import sqlite3
import time
from types import SimpleNamespace as NS

from cryptography.fernet import Fernet
import pytest
from qdrant_client import QdrantClient, models

from eigenlens.config import Config
from eigenlens.connections import ConnectionStore
from eigenlens.databases import CUSTOM, register_adapter
from eigenlens.databases.base import Adapter, ConnectionOptions, record
from eigenlens.schemas import Record, VTError
from eigenlens.service import Service
from test_density import fixture_data, operation


class FixtureAdapter(Adapter):
    rows = [Record(id=str(i), vector=[1., float(i + 1)], text=f"cash withdrawal {i}") for i in range(6)]
    closed = 0
    streams_closed = 0

    def list_collections(self):
        return ["questions"]

    def describe(self, collection):
        return {"dimension": 2, "vector_field": "body"}

    def scan(self, collection, batch_size):
        try:
            yield from self.rows
        finally:
            type(self).streams_closed += 1

    def search(self, collection, vector, limit):
        yield from self.rows[:limit]

    def close(self):
        type(self).closed += 1


@pytest.fixture
def database_service(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENLENS_CONNECTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setitem(CUSTOM, "fixture", FixtureAdapter)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data_dir": str(tmp_path / "data"), "callers": {"alice": {}, "bob": {}, "local": {}}}))
    service = Service(Config(str(path)))
    try:
        yield service
    finally:
        service.close()


def connection(service, **kwargs):
    return service.connect_database("alice", "fixture", "https://db.example", "private-api-key", **kwargs)["connection_id"]


def datasets_count(service):
    with service.store.connect() as db:
        return db.execute("SELECT COUNT(*) FROM objects WHERE kind='ds'").fetchone()[0]


def test_encrypted_persistence_and_owner_isolation(database_service):
    service = database_service
    cid = connection(service, name="Banking")
    assert service.list_connections("alice")["connections"][0]["name"] == "Banking"
    assert service.list_connections("bob")["count"] == 0
    for method, args in [(service.list_collections, (cid,)),
                         (service.retrieve_records, (cid, "questions")),
                         (service.disconnect_database, (cid,))]:
        with pytest.raises(VTError, match="unauthorized"):
            method("bob", *args)
    with service.databases.store.db() as db:
        row = db.execute("SELECT * FROM connections").fetchone()
        assert b'private-api-key' not in row['encrypted'] and b'db.example' not in row['encrypted']
    for path in service.config.root.glob('connections.sqlite*'):
        assert b'private-api-key' not in path.read_bytes()
        assert b'db.example' not in path.read_bytes()
    reopened = ConnectionStore(service.config.root)
    assert reopened.get("alice", cid)["api_key"] == "private-api-key"
    assert "private-api-key" not in json.dumps(service.list_connections("alice"))
    assert service.list_collections("alice", cid)["collections"] == ["questions"]
    assert service.disconnect_database("alice", cid)["status"] == "disconnected"
    with pytest.raises(VTError, match="unavailable"):
        reopened.get("alice", cid)


def test_missing_wrong_and_changed_encryption_keys(database_service, monkeypatch):
    service = database_service
    original = __import__('os').environ['EIGENLENS_CONNECTION_KEY']
    monkeypatch.delenv("EIGENLENS_CONNECTION_KEY")
    with pytest.raises(VTError, match="Set EIGENLENS_CONNECTION_KEY"):
        connection(service)
    assert service.list_connections("alice")["count"] == 0
    monkeypatch.setenv("EIGENLENS_CONNECTION_KEY", "bad-key")
    with pytest.raises(VTError, match="valid Fernet"):
        connection(service)
    monkeypatch.setenv("EIGENLENS_CONNECTION_KEY", original)
    cid = connection(service)
    monkeypatch.setenv("EIGENLENS_CONNECTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(VTError, match="original"):
        service.list_collections("alice", cid)


def test_complete_selection_and_retrieved_snapshot(database_service):
    service, cid = database_service, connection(database_service)
    all_rows = service.retrieve_records("alice", cid, "questions")
    assert all_rows["count"] == 6 and all_rows["scope"] == "complete_selection"
    assert all_rows["selection"]["mode"] == "all"
    subset = service.retrieve_records("alice", cid, "questions", query_vector=[1, 2], search_limit=3)
    assert subset["count"] == 3 and subset["scope"] == "retrieval_results"
    assert subset["selection"]["mode"] == "top_k" and subset["selection"]["search_limit"] == 3
    spec = service.store.get("alice", subset["dataset_id"], "ds")
    assert spec["source"]["db_type"] == "fixture" and len(spec["records"]) == 3
    service.disconnect_database("alice", cid)
    assert service.store.get("alice", subset["dataset_id"], "ds")["records"] == spec["records"]


@pytest.mark.parametrize("failure", ["rows", "bytes", "duplicates", "missing_vector", "dimensions", "models", "revisions"])
def test_retrieval_fails_without_partial_snapshot(database_service, monkeypatch, failure):
    service = database_service
    rows = [r.model_copy(deep=True) for r in FixtureAdapter.rows]
    if failure == "rows":
        service.config.limits.max_rows = 2
    elif failure == "bytes":
        service.config.limits.max_bytes = 1024
        rows[0].text = 'x' * 1500
    elif failure == "duplicates":
        rows[1].id = rows[0].id
    elif failure == "missing_vector":
        rows[1].vector = None
    elif failure == "dimensions":
        rows[1].vector = [1., 2., 3.]
    elif failure == "models":
        rows[0].metadata['embedding_model'] = 'model-one'
        rows[1].metadata['embedding_model'] = 'model-two'
    elif failure == "revisions":
        rows[0].metadata['embedding_revision'] = 'revision-one'
        rows[1].metadata['embedding_revision'] = 'revision-two'
    monkeypatch.setattr(FixtureAdapter, 'rows', rows)
    cid = connection(service)
    before, closed = datasets_count(service), FixtureAdapter.streams_closed
    with pytest.raises(VTError):
        service.retrieve_records("alice", cid, "questions", search_limit=min(1000, service.config.limits.max_rows))
    assert datasets_count(service) == before
    assert FixtureAdapter.streams_closed == closed + 1


def test_query_model_resolution_and_invalid_vectors(database_service, monkeypatch):
    service, cid = database_service, connection(database_service)
    with pytest.raises(VTError, match="unknown"):
        service.retrieve_records("alice", cid, "questions", query="ATM")
    rows = [r.model_copy(deep=True) for r in FixtureAdapter.rows]
    for item in rows:
        item.metadata.update(embedding_model="sentence-transformers/test-model", embedding_revision="snapshot")
    monkeypatch.setattr(FixtureAdapter, 'rows', rows)
    seen = []
    def embed(text, model, revision, normalize):
        seen.append((text, model, revision, normalize))
        return [1., 2.]
    monkeypatch.setattr("eigenlens.retrieval.embed_query", embed)
    result = service.retrieve_records("alice", cid, "questions", query="ATM", search_limit=3)
    assert result["count"] == 3
    assert seen == [("ATM", "sentence-transformers/test-model", "snapshot", True)]
    for vector in ([0, 0], [float('nan'), 1], [1, 2, 3], []):
        with pytest.raises(VTError):
            service.retrieve_records("alice", cid, "questions", query_vector=vector)
    with pytest.raises(VTError, match="not both"):
        service.retrieve_records("alice", cid, "questions", query="ATM", query_vector=[1, 2])


def test_operator_default_is_local_only(database_service, monkeypatch):
    service = database_service
    monkeypatch.setenv("EIGENLENS_DATABASE_TYPE", "fixture")
    monkeypatch.setenv("EIGENLENS_DATABASE_URL", "https://default.example")
    monkeypatch.setenv("EIGENLENS_DATABASE_API_KEY", "operator-secret")
    result = service.connect_database("local")
    assert result["name"] == "default"
    assert service.databases.store.get("local", result["connection_id"])["api_key"] == "operator-secret"
    with pytest.raises(VTError, match="Supply db_type"):
        service.connect_database("alice")
    cid = connection(service)
    assert service.databases.store.get("alice", cid)["api_key"] == "private-api-key"


def test_connection_limits_unknown_types_and_error_redaction(database_service, monkeypatch):
    service = database_service
    service.config.limits.max_connections = 1
    connection(service)
    with pytest.raises(VTError, match="Connection limit"):
        connection(service)
    with pytest.raises(VTError, match="Unsupported"):
        service.connect_database("alice", "unknown")
    class Broken(FixtureAdapter):
        def list_collections(self):
            raise RuntimeError("request failed with private-api-key and https://user:password@db")
    monkeypatch.setitem(CUSTOM, "fixture", Broken)
    with pytest.raises(VTError) as caught:
        service.connect_database("bob", "fixture", "https://db.example", "private-api-key")
    assert "private-api-key" not in str(caught.value) and 'password' not in str(caught.value)
    assert service.list_connections("bob")["count"] == 0


def test_collection_options_inherit_connection_settings():
    adapter = FixtureAdapter({"options": {"embedding_model": "model", "text_field": "body_text",
        "collections": {"questions": {"vector_field": "body"}}}})
    opts = adapter.collection_options('questions')
    assert opts.vector_field == 'body' and opts.embedding_model == 'model' and opts.text_field == 'body_text'


def test_real_qdrant_pagination_search_and_analysis(database_service, tmp_path, monkeypatch):
    from eigenlens.databases import qdrant
    data_path = str(tmp_path / 'qdrant')
    fixtures = fixture_data().records
    with closing(QdrantClient(path=data_path)) as client:
        client.create_collection('questions', vectors_config={'body':models.VectorParams(size=len(fixtures[0].vector), distance=models.Distance.COSINE)})
        client.upsert('questions', points=[models.PointStruct(id=i, vector={'body':r.vector}, payload={'text':r.text}) for i,r in enumerate(fixtures)])
    monkeypatch.setattr(qdrant, 'QdrantClient', lambda **kwargs: QdrantClient(path=data_path))
    service = database_service
    service.config.limits.batch_size = 7
    cid = service.connect_database('alice', 'qdrant', 'http://localhost:6333')['connection_id']
    assert service.list_collections('alice', cid)['collections'] == ['questions']
    retrieved = service.retrieve_records('alice', cid, 'questions')
    assert retrieved['count'] == 33
    selected = service.retrieve_records('alice', cid, 'questions', query_vector=fixtures[0].vector, search_limit=9)
    assert selected['count'] == 9 and selected['scope'] == 'retrieval_results'
    job = service.run_analysis('alice', retrieved['dataset_id'], [operation(), {'id':'themes','type':'topics','clusters_from':'groups'}])
    deadline = time.monotonic() + 90  # Includes cold BERTopic imports in the spawned process.
    while job['status'] not in {'succeeded','failed','cancelled'}:
        assert time.monotonic() < deadline, job
        time.sleep(.05)
        job = service.get_job('alice', job['job_id'])
    assert job['status'] == 'succeeded', job
    summary = service.get_summary('alice', job['results']['groups'])
    assert summary['processed'] == 33 and summary['source']['db_type'] == 'qdrant'
    assert summary['source']['collection'] == 'questions'
    assert service.get_result_page('alice', job['results']['themes'])['rows']


def test_pinecone_pagination_fetch_and_query_mapping(monkeypatch):
    pytest.importorskip('pinecone')
    from eigenlens.databases import pinecone
    class Index:
        def list_paginated(self, **kwargs):
            second = kwargs.get('pagination_token') == 'page2'
            return NS(vectors=[NS(id='2' if second else '1')], pagination=NS(next=None if second else 'page2'))
        def fetch(self, ids, namespace, **kwargs):
            return NS(vectors={identifier:NS(values=[1,2], metadata={'content':'ATM'}) for identifier in ids})
        def query(self, **kwargs):
            assert kwargs['include_values'] and kwargs['include_metadata']
            return NS(matches=[NS(id='1', values=[1,2], metadata={'content':'ATM'})])
        def close(self): pass
    class Client:
        def get(self, path):
            result = {'indexes':[{'name':'index','host':'index.example'}]} if path == '/indexes' else {'host':'index.example','dimension':2}
            return NS(raise_for_status=lambda:None,json=lambda:result)
        def close(self): pass
    monkeypatch.setattr(pinecone.httpx, 'Client', lambda **kwargs: Client())
    monkeypatch.setattr(pinecone, 'Index', lambda **kwargs: Index())
    adapter = pinecone.PineconeAdapter({'api_key':'test','options':{'text_field':'content'}})
    try:
        assert adapter.list_collections() == ['index']
        assert [r.id for r in adapter.scan('index', 10)] == ['1','2']
        assert list(adapter.search('index', [1,2], 1))[0].text == 'ATM'
    finally: adapter.close()


def test_milvus_iterator_cleanup_and_vector_mapping(monkeypatch):
    pytest.importorskip('pymilvus')
    from eigenlens.databases import milvus
    closed = []
    class Iterator:
        def __init__(self): self.page = 0
        def next(self):
            self.page += 1
            return [{'pk':self.page,'body':[1,2],'text':'ATM'}] if self.page <= 2 else []
        def close(self): closed.append('iterator')
    class Client:
        def list_collections(self): return ['questions']
        def describe_collection(self, **kwargs):
            return {'fields':[{'name':'pk','type':5,'is_primary':True}, {'name':'body','type':101,'params':{'dim':2}}, {'name':'text','type':21}]}
        def query_iterator(self, **kwargs):
            assert 'body' in kwargs['output_fields']
            return Iterator()
        def search(self, **kwargs):
            return [[{'id':1,'entity':{'body':[1,2],'text':'ATM'}}]]
        def close(self): closed.append('client')
    monkeypatch.setattr(milvus, 'MilvusClient', lambda **kwargs: Client())
    adapter = milvus.MilvusAdapter({'url':'http://localhost:19530'})
    assert [r.id for r in adapter.scan('questions', 1)] == ['1','2']
    stream = adapter.scan('questions', 1)
    next(stream); stream.close()
    assert closed == ['iterator','iterator']
    assert list(adapter.search('questions',[1,2],1))[0].id == '1'
    adapter.close()


@pytest.mark.parametrize('field',['body','default'])
def test_weaviate_named_vector_mapping(monkeypatch, field):
    pytest.importorskip('weaviate')
    from eigenlens.databases import weaviate
    point = NS(uuid='uuid',vector={field:[1,2]},properties={'content':'ATM'})
    def near_vector(**kwargs):
        assert kwargs['target_vector'] == field and kwargs['include_vector']
        return NS(objects=[point])
    collection = NS(config=NS(get=lambda:NS(vector_config={field:None,'unused':None})),
        iterator=lambda **kwargs:iter([point]), query=NS(near_vector=near_vector))
    client = NS(collections=NS(use=lambda name:collection, list_all=lambda **kwargs:{'Questions':None}),close=lambda:None)
    monkeypatch.setattr(weaviate.weaviate,'connect_to_custom',lambda **kwargs:client)
    adapter = weaviate.WeaviateAdapter({'url':'http://localhost:8080','options':{'text_field':'content','vector_field':field}})
    assert adapter.list_collections() == ['Questions']
    assert list(adapter.scan('Questions',1))[0].text == 'ATM'
    assert list(adapter.search('Questions',[1,2],1))[0].vector == [1.,2.]
    adapter.close()


def test_pgvector_identifiers_bound_values_and_read_only(monkeypatch):
    pytest.importorskip('psycopg')
    from eigenlens.databases import pgvector
    calls = []
    table = 'records"; DROP TABLE records; --'
    rows = [{'_id':'1','_vector':'[1,2]','_payload':{'text':'ATM'}}]
    class Cursor:
        def __init__(self): self.read = False
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def execute(self, statement, params=None): calls.append((statement.as_string(), params))
        def fetchmany(self, size):
            if self.read: return []
            self.read = True; return rows
        def __iter__(self): return iter(rows)
    class Client:
        read_only = False
        def execute(self, statement, params):
            assert params == ('public',)
            return NS(fetchall=lambda:[{'name':table,'field':'body','dimension':2}])
        def cursor(self, **kwargs): return Cursor()
        def close(self): pass
    client = Client()
    def connect(url, **kwargs):
        assert 'default_transaction_read_only=on' in kwargs['options']
        return client
    monkeypatch.setattr(pgvector.psycopg,'connect',connect)
    adapter = pgvector.PgvectorAdapter({'url':'postgresql://user:password@localhost/db'})
    assert client.read_only
    assert list(adapter.scan(table,2))[0].text == 'ATM'
    assert list(adapter.search(table,[1,2],1))[0].id == '1'
    assert '"public"."records""; DROP TABLE records; --"' in calls[0][0]
    assert calls[1][1] == ('[1, 2]', 1) and '%s::vector LIMIT %s' in calls[1][0]
    adapter.close()


def test_http_database_tools_and_caller_isolation(database_service, monkeypatch):
    from starlette.testclient import TestClient
    from eigenlens.server import AccessMiddleware, build_server
    service = database_service
    service.config.callers = {'alice':{'token_env':'ALICE_TOKEN'}, 'bob':{'token_env':'BOB_TOKEN'}}
    monkeypatch.setenv('ALICE_TOKEN','alice-test-token')
    monkeypatch.setenv('BOB_TOKEN','bob-test-token')
    server = build_server(service)
    with TestClient(AccessMiddleware(server.streamable_http_app(), service.config)) as client:
        counter = 0
        def rpc(method, params, token='alice-test-token'):
            nonlocal counter
            counter += 1
            response = client.post('/mcp',json={'jsonrpc':'2.0','id':counter,'method':method,'params':params},
                headers={'Authorization':'Bearer '+token,'Accept':'application/json, text/event-stream'})
            assert response.status_code == 200, response.text
            return response.json()['result']
        rpc('initialize',{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'db-test','version':'1'}})
        def call(name, args, token='alice-test-token'):
            result = rpc('tools/call', {'name':name,'arguments':args},token)
            assert not result.get('isError'), result
            return result.get('structuredContent') or json.loads(result['content'][0]['text'])
        created = call('connect_database',{'db_type':'fixture','url':'https://db.example','api_key':'private-api-key'})
        cid = created['connection_id']
        assert 'private-api-key' not in json.dumps(created)
        assert call('list_collections',{'connection_id':cid})['collections'] == ['questions']
        dataset = call('retrieve_records',{'connection_id':cid,'collection':'questions'})
        assert dataset['count'] == 6 and dataset['scope'] == 'complete_selection'
        assert call('list_connections',{},'bob-test-token')['count'] == 0
        denied = rpc('tools/call',{'name':'retrieve_records','arguments':{'connection_id':cid,'collection':'questions'}},'bob-test-token')
        assert denied['isError'] and 'unauthorized' in json.dumps(denied)
        call('disconnect_database',{'connection_id':cid})
        assert call('list_connections',{})['count'] == 0


def test_full_scan_ignores_search_limit_and_cleans_up_at_row_limit(database_service):
    service, cid = database_service, connection(database_service)
    service.config.limits.max_rows = 2
    before = datasets_count(service)
    with pytest.raises(VTError, match='row limits'):
        service.retrieve_records('alice',cid,'questions')
    assert datasets_count(service) == before


def test_retrieval_timeout_creates_no_snapshot(database_service, monkeypatch):
    service, cid = database_service, connection(database_service)
    ticks = iter([0, 1000])
    monkeypatch.setattr('eigenlens.retrieval.time',NS(monotonic=lambda:next(ticks)))
    with pytest.raises(VTError, match='max_seconds'):
        service.retrieve_records('alice',cid,'questions')
    assert datasets_count(service) == 0


def test_adapter_registration_and_secret_options(database_service, monkeypatch):
    service = database_service
    monkeypatch.setitem(CUSTOM,'customtest',FixtureAdapter)
    register_adapter('customtest',FixtureAdapter)
    with pytest.raises(VTError, match='unique'):
        register_adapter('qdrant',FixtureAdapter)
    created = service.connect_database('alice','customtest','https://db.example',options={'adapter_options':{'private':'plugin-secret'}})
    spec = service.databases.store.get('alice',created['connection_id'])
    assert spec['options']['adapter_options']['private'] == 'plugin-secret'
    assert 'plugin-secret' not in service.databases.store.path.read_bytes().decode(errors='ignore')


def test_metadata_connection_closes_and_rolls_back(database_service):
    service = database_service
    with service.store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM objects').fetchone()[0] == 0
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        db.execute('SELECT 1')
    with pytest.raises(RuntimeError):
        with service.databases.store.db() as connections:
            connections.execute("INSERT INTO connections VALUES ('temporary','alice','fixture','test',X'00','now')")
            raise RuntimeError('rollback transaction')
    assert service.list_connections('alice')['count'] == 0


def test_embedding_failure_is_actionable_and_redacted(database_service, monkeypatch):
    service = database_service
    cid = connection(service, options={'embedding_model':'private-model'})
    def fail(*args):
        raise RuntimeError('private-model-access-token')
    monkeypatch.setattr('eigenlens.retrieval.embed_query',fail)
    with pytest.raises(VTError, match='original embedding provider') as caught:
        service.retrieve_records('alice',cid,'questions',query='ATM')
    assert 'private-model-access-token' not in str(caught.value)
    assert datasets_count(service) == 0
