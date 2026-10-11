import json
import pytest
from starlette.testclient import TestClient
from vector_tools.config import Config
from vector_tools.service import Service
from vector_tools.server import build_server, AccessMiddleware
from vector_tools.schemas import VTError


@pytest.fixture
def uploads(tmp_path,monkeypatch):
    monkeypatch.setenv('VECTOR_ML_MCP_TOKEN','upload-test-token')
    monkeypatch.setenv('OTHER_TOKEN','other-upload-token')
    monkeypatch.delenv('VECTOR_ML_DATA',raising=False)
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps({'data_dir':str(tmp_path/'data'),
        'limits':{'max_bytes':4*1024*1024},'callers':{
            'local':{'token_env':'VECTOR_ML_MCP_TOKEN'},
            'other':{'token_env':'OTHER_TOKEN'}}}))
    config=Config(str(config_path));service=Service(config)
    try:
        server=build_server(service)
        with TestClient(AccessMiddleware(server.streamable_http_app(),config)) as client:
            yield client,service,{'Authorization':'Bearer upload-test-token','Content-Type':'application/x-ndjson'}
    finally:service.close()


def test_upload_over_rpc_limit_registers_complete_selection(uploads):
    client,service,headers=uploads
    content=json.dumps({'id':'a','vector':[1,0],'text':'x'*(2*1024*1024+100)})+'\n'
    response=client.post('/ingress/large.jsonl',content=content,headers=headers)
    assert response.status_code==201
    dataset=service.register_records('local',artifact_name='large.jsonl',selection_scope='complete_selection')
    assert dataset['count']==1 and dataset['scope']=='complete_selection'
    assert client.delete('/ingress/large.jsonl',headers=headers).status_code==200
    assert not (service.config.root/'ingress/local/large.jsonl').exists()
    # The larger upload allowance does not change the 2 MiB MCP RPC limit.
    assert client.post('/mcp',content=content,headers=headers).status_code==413


def test_upload_authentication_and_caller_isolation(uploads):
    client,service,headers=uploads
    content=json.dumps({'id':'a','vector':[1,0]})+'\n'
    assert client.post('/ingress/input.jsonl',content=content).status_code==401
    assert client.post('/ingress/input.jsonl',content=content,headers=headers).status_code==201
    with pytest.raises(VTError,match='unavailable'):
        service.register_records('other',artifact_name='input.jsonl')
    assert client.delete('/ingress/input.jsonl',headers={'Authorization':'Bearer other-upload-token'}).status_code==200
    assert (service.config.root/'ingress/local/input.jsonl').is_file()


def test_upload_rejects_paths_overwrites_and_incomplete_inputs(uploads):
    client,service,headers=uploads
    assert client.post('/ingress/..jsonl',content='{}',headers=headers).status_code==400
    assert client.post('/ingress/input.txt',content='{}',headers=headers).status_code==400
    assert client.post('/ingress/input.jsonl',content='original',headers=headers).status_code==201
    assert client.post('/ingress/input.jsonl',content='replacement',headers=headers).status_code==409
    assert (service.config.root/'ingress/local/input.jsonl').read_text()=='original'
    def chunks():
        for _ in range(5):yield b'x'*(1024*1024)
    assert client.post('/ingress/too-large.jsonl',content=chunks(),headers=headers).status_code==413
    folder=service.config.root/'ingress/local'
    assert not (folder/'too-large.jsonl').exists()
    assert not list(folder.glob('.upload-*'))
