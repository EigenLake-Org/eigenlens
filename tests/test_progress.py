"""Detailed job information is durable, owner-scoped and independent of status."""
import json
import sqlite3
import pytest
import eigenlens as vt
from eigenlens.artifacts import Store, now
from eigenlens.execution_context import progress_reporter, report_stage
from eigenlens.schemas import VTError
from test_density import fixture_data, operation
from test_service import service, wait


def insert_job(store, id='job_test', status='running'):
    with store.connect() as db:
        db.execute('INSERT INTO jobs(id,owner,status,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                   (id, 'local', status, '{}', now(), now()))


@pytest.mark.parametrize('outcome', ['failed', 'cancelled'])
def test_terminal_status_retains_last_stage_and_rejects_late_updates(tmp_path, outcome):
    store=Store(tmp_path)
    insert_job(store)
    store.update_stage('job_test', dict(stage='dbscan_search', message='Evaluating parameters',
                                      completed=12, total=42, unit='trials'))
    before=store.job('local','job_test')['info']
    store.update_job('job_test',outcome,error='Stopped')
    store.update_stage('job_test', dict(stage='pca',message='Late callback'))
    store.update_job('job_test','running')
    persisted=Store(tmp_path).job('local','job_test')
    assert persisted['status']==outcome and persisted['info']==before
    assert persisted['error']=='Stopped'
    with pytest.raises(VTError,match='unauthorized'):
        store.job('other','job_test')


def test_existing_database_migrates_without_losing_jobs(tmp_path):
    with sqlite3.connect(tmp_path/'metadata.sqlite') as db:
        db.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY,owner TEXT,status TEXT,payload TEXT,results TEXT DEFAULT '{}',error TEXT,created_at TEXT,updated_at TEXT,idempotency_key TEXT,request_hash TEXT)")
        db.execute("INSERT INTO jobs(id,owner,status,payload,created_at,updated_at) VALUES('old','local','succeeded','{}','before','before')")
    assert Store(tmp_path).job('local','old')['info']=={}


def test_algorithm_stages_and_real_counters_do_not_change_results():
    data=fixture_data()
    ops=[operation(), {'id':'themes','type':'topics','clusters_from':'groups'}]
    events=[]
    with progress_reporter(events.append):
        results=vt.analyze(data,ops)
    stages={event['stage'] for event in events}
    assert {'validating_vectors','normalizing_vectors','pca','neighbor_graph','dbscan_search',
            'summarizing_clusters','loading_bertopic','bertopic','summarizing_topics'}<=stages
    trials=[event for event in events if event['stage']=='dbscan_search']
    assert [event['completed'] for event in trials]==[0,1,2]
    assert all(event['total']==len(results['groups'].summary_data['parameter_search'])==2 for event in trials)
    assert all(event['operation_id']=='groups' and event['operation_index']==1 and event['operation_count']==2 for event in trials)
    neighbors=[event for event in events if event['stage']=='neighbor_graph']
    assert neighbors[-1]['completed']==neighbors[-1]['total']==33
    assert next(event for event in events if event['stage']=='bertopic')['operation_id']=='themes'
    assert len(results['groups'].rows)==33 and results['themes'].processed==33
    # Reporter and operation context must not leak to subsequent SDK calls.
    report_stage('outside','Outside execution')
    assert events[-1]['stage']=='summarizing_topics'
    assert 'card payment' not in json.dumps(events)


def test_get_job_preserves_contract_and_reports_cache_and_completion(service):
    ds=service.register_records('local',records=[r.model_dump() for r in fixture_data().records])
    first=service.run_analysis('local',ds['dataset_id'],[operation()])
    assert first['info']['stage'] and first['info']['message']
    assert service.get_capabilities('local')['execution']['detailed_job_stages']
    done=wait(service,first['job_id'])
    assert done['status']=='succeeded' and done['info']['stage']=='completed'
    assert done['progress']=={'phase':'succeeded','fraction':1}
    cached=service.run_analysis('local',ds['dataset_id'],[operation()])
    assert cached['status']=='succeeded' and cached['info']['stage']=='reusing_results'
    assert cached['cache_hit'] and cached['results']==done['results']


def test_get_job_exposes_persisted_stage_without_changing_status(service):
    insert_job(service.store)
    service.store.update_stage('job_test',dict(stage='pca',message='Reducing dimensions',records=33))
    job=service.get_job('local','job_test')
    assert job['status']=='running' and job['progress']=={'phase':'running','fraction':None}
    assert job['info']['stage']=='pca' and job['info']['records']==33
    service.store.update_job('job_test','failed',error='Interrupted')
    assert service.get_job('local','job_test')['info']==job['info']
    with pytest.raises(VTError):
        service.get_job('other','job_test')


def test_real_worker_reports_stage_before_cancel(service):
    from worker_helpers import paused_pca_worker, wait_started
    service.supervisor.worker_target=paused_pca_worker
    ds=service.register_records('local',records=[r.model_dump() for r in fixture_data().records])
    job=service.run_analysis('local',ds['dataset_id'],[operation()])
    wait_started(service.config.root,job['job_id'])
    running=service.get_job('local',job['job_id'])
    assert running['status']=='running' and running['info']['stage']=='pca'
    assert running['info']['operation_id']=='groups' and running['info']['records']==33
    cancelled=service.cancel_job('local',job['job_id'])
    assert cancelled['status']=='cancelled' and cancelled['info']==running['info']
