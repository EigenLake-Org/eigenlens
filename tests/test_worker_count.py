import json
import time
import pytest
from vector_tools.config import Config
from vector_tools.service import Service
from vector_tools.schemas import VTError


@pytest.mark.parametrize('workers',[1,2,16])
def test_worker_override_and_shared_memory_budget(tmp_path,monkeypatch,workers):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'limits':{'max_jobs':1,'max_working_bytes':8*1024**3}}))
    monkeypatch.setenv('VECTOR_ML_WORKERS',str(workers))
    monkeypatch.setenv('VECTOR_ML_TOTAL_WORKING_BYTES',str(8*1024**3))
    config=Config(str(path))
    assert config.limits.max_jobs==workers
    assert config.limits.max_working_bytes==(8*1024**3)//workers
    # Startup overrides never rewrite the persisted configuration.
    assert json.loads(path.read_text())['limits']['max_jobs']==1


@pytest.mark.parametrize('value',['0','17','-1','1.5','two','', '01'])
def test_invalid_worker_count_is_rejected(monkeypatch,value):
    monkeypatch.setenv('VECTOR_ML_WORKERS',value)
    with pytest.raises(VTError,match='1 to 16'):Config()


def test_configured_default_is_preserved(monkeypatch):
    monkeypatch.delenv('VECTOR_ML_WORKERS',raising=False)
    monkeypatch.delenv('VECTOR_ML_TOTAL_WORKING_BYTES',raising=False)
    assert Config('config/chat-worker.json').limits.max_jobs==1


def test_two_real_jobs_run_and_third_waits(tmp_path,monkeypatch):
    from vector_tools.demo import tickets
    from worker_helpers import paused_cluster_worker, wait_started
    monkeypatch.setenv('VECTOR_ML_DATA',str(tmp_path))
    monkeypatch.setenv('VECTOR_ML_WORKERS','2')
    monkeypatch.setenv('VECTOR_ML_TOTAL_WORKING_BYTES',str(128*1024**2))
    service=Service()
    service.supervisor.worker_target=paused_cluster_worker
    try:
        ds=service.register_records('local',records=[r.model_dump() for r in tickets()])
        ops=[{'id':'groups','type':'cluster','algorithm':'dbscan_search'},
             {'id':'themes','type':'topics','clusters_from':'groups'}]
        jobs=[service.run_analysis('local',ds['dataset_id'],ops)['job_id'] for _ in range(3)]
        for job_id in jobs[:2]:wait_started(tmp_path, job_id)
        assert [service.get_job('local',id)['status'] for id in jobs]==['running','running','queued']
        with service.supervisor.mutex:
            assert len(service.supervisor.running)==2
            assert len({p.pid for p,_,_ in service.supervisor.running.values()})==2
        assert service.cancel_job('local',jobs[0])['status']=='cancelled'
        assert service.get_job('local',jobs[1])['status']=='running'
        (tmp_path/'release').touch()
        deadline=time.monotonic()+90
        for job_id in jobs[1:]:
            while (job:=service.get_job('local',job_id))['status'] not in {'succeeded','failed','cancelled'}:
                assert time.monotonic()<deadline
                time.sleep(.05)
            assert job['status']=='succeeded', job
            assert service.get_summary('local',job['results']['groups'])['processed']==62
            assert service.get_summary('local',job['results']['themes'])['processed']==62
    finally:service.close()
