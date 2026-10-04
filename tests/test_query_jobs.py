import json
import time
from pathlib import Path
from starlette.testclient import TestClient
from vector_tools.artifacts import Store
from examples.query_job_server import create_app


def fake_worker(root, config_path, job_id, owner):
    store=Store(root)
    store.update_job(job_id,'running')
    payload=store.job(owner,job_id)['payload']
    if payload['query']=='slow':
        time.sleep(30)
    output=Path(root)/'plots'/job_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'banking77_dbscan_3d.html').write_text('<html>verified</html>')
    ref=store.put(owner,'query',{'retrieved':5,'groups':[{'queries':5}]})
    store.update_job(job_id,'succeeded',{'summary_id':ref})


def config(tmp_path, max_seconds=10):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'data_dir':str(tmp_path/'jobs'),'limits':{'max_jobs':1,'max_pending_jobs':2,'max_seconds':max_seconds}}))
    return str(path)


def wait(client, job_id):
    for _ in range(200):
        job=client.get('/jobs/'+job_id).json()
        if job['status'] in {'succeeded','failed','cancelled'}:
            return job
        time.sleep(.05)
    raise AssertionError('timeout')


def test_auth_validation_idempotency_results_and_restart(tmp_path):
    path=config(tmp_path)
    app=create_app(path,token='test',worker_target=fake_worker)
    with TestClient(app) as client:
        assert client.post('/jobs',json={'query':'x'}).status_code==401
        assert client.get('/health').status_code==200
        client.headers['Authorization']='Bearer test'
        assert client.post('/jobs',json={'query':'x','top_k':2001}).status_code==422
        assert client.post('/jobs',content=b'x'*17000).status_code==413
        response=client.post('/jobs',json={'query':'atm'},headers={'Idempotency-Key':'one'})
        assert response.status_code==202
        job_id=response.json()['job_id']
        assert client.post('/jobs',json={'query':'atm'},headers={'Idempotency-Key':'one'}).json()['job_id']==job_id
        assert client.post('/jobs',json={'query':'other'},headers={'Idempotency-Key':'one'}).status_code==409
        job=wait(client,job_id)
        assert job['status']=='succeeded',job
        assert job['summary']['retrieved']==5
        assert client.get(job['artifacts']['plot']).text=='<html>verified</html>'
        assert client.get('/jobs/missing').status_code==404
        assert client.get(f'/jobs/{job_id}/secret').status_code==404
    with TestClient(create_app(path,token='test',worker_target=fake_worker)) as client:
        client.headers['Authorization']='Bearer test'
        assert client.get('/jobs/'+job_id).json()['status']=='succeeded'
        assert client.get(f'/jobs/{job_id}/plot').status_code==200


def test_cancel_running_job(tmp_path):
    with TestClient(create_app(config(tmp_path),token='test',worker_target=fake_worker)) as client:
        client.headers['Authorization']='Bearer test'
        job_id=client.post('/jobs',json={'query':'slow'}).json()['job_id']
        for _ in range(100):
            if client.get('/jobs/'+job_id).json()['status']=='running':
                break
            time.sleep(.05)
        assert client.delete('/jobs/'+job_id).json()['status']=='cancelled'
        assert client.get(f'/jobs/{job_id}/plot').status_code==409


def test_worker_timeout(tmp_path):
    with TestClient(create_app(config(tmp_path,1),token='test',worker_target=fake_worker)) as client:
        client.headers['Authorization']='Bearer test'
        job_id=client.post('/jobs',json={'query':'slow'}).json()['job_id']
        job=wait(client,job_id)
        assert job['status']=='failed'
        assert 'max_seconds' in job['error']
