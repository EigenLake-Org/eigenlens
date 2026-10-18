import time
import pytest
from eigenlens.service import Service
from eigenlens.schemas import VTError
from eigenlens.demo import tickets

def wait(service, id):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        j = service.get_job("local", id)
        if j["status"] in {"succeeded", "failed", "cancelled"}:
            return j
        time.sleep(.05)
    raise AssertionError("job timeout")

@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENLENS_DATA", str(tmp_path))
    service = Service()
    yield service
    service.close()

def test_durable_jobs_idempotency_access_pages(service):
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()[:30]])
    ops = [{"id":"c", "type":"cluster", "parameters":{"eps_values":[.3,.5]}}]
    job = service.run_analysis("local", ds["dataset_id"], ops, idempotency_key="test")
    duplicate = service.run_analysis("local", ds["dataset_id"], ops, idempotency_key="test")
    assert duplicate["job_id"] == job["job_id"]
    with pytest.raises(VTError, match="different"):
        service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster", "parameters":{"eps_values":[.4]}}], idempotency_key="test")
    j = wait(service, job["job_id"])
    assert j["status"] == "succeeded", j
    ref = j["results"]["c"]
    assert service.get_summary("local", ref)["processed"] == 30
    page = service.get_result_page("local", ref, limit=7)
    ids = [r["id"] for r in page["rows"]]
    while page["next_cursor"]:
        page = service.get_result_page("local", ref, page["next_cursor"], limit=7)
        ids += [r["id"] for r in page["rows"]]
    assert len(set(ids)) == 30
    with pytest.raises(VTError, match="unauthorized"):
        service.get_summary("other", ref)
    with pytest.raises(VTError, match="unauthorized"):
        service.get_job("other", j["job_id"])
    from eigenlens.artifacts import Store
    assert Store(service.config.root).get("local", ref, "res")["processed"] == 30

def test_failed_cancelled_and_artifact(service):
    ds = service.register_records("local", records=[])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster"}])
    assert wait(service, job["job_id"])["status"] == "failed"
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"c","type":"cluster"}])
    result = service.cancel_job("local", job["job_id"])
    assert result["status"] in {"cancelled", "succeeded"}
    assert job["job_id"] not in service.supervisor.running
    folder = service.config.root / "ingress" / "local"
    folder.mkdir()
    (folder / "records.jsonl").write_text('\n'.join(r.model_dump_json() for r in tickets()))
    assert service.register_records("local", artifact_name="records.jsonl")["count"] == 62
    with pytest.raises(VTError):
        service.register_records("local", artifact_name="../records.jsonl")

def test_restart_marks_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENLENS_DATA", str(tmp_path))
    from eigenlens.artifacts import Store, now
    store = Store(tmp_path)
    with store.connect() as db:
        db.execute("INSERT INTO jobs(id,owner,status,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)", ("job_interrupted", "local", "running", "{}", now(), now()))
    service = Service()
    try:
        assert service.get_job("local", "job_interrupted")["status"] == "failed"
        with pytest.raises(VTError, match="supervisor"):
            Service()
    finally:
        service.close()

def test_registration_scope_and_persistent_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("EIGENLENS_DATA", str(tmp_path))
    service = Service()
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    assert ds["scope"] == "retrieval_results"
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"s", "type":"cluster", "parameters":{"eps_values":[.3,.5]}}])
    done = wait(service, job["job_id"])
    assert done["status"] == "succeeded"
    ref = done["results"]["s"]
    before = service.get_summary("local", ref)
    service.close()
    restarted = Service()
    try:
        assert restarted.get_summary("local", ref) == before
        assert restarted.get_job("local", job["job_id"])["results"]["s"] == ref
        assert before["scope"] == "retrieval_results"
    finally:
        restarted.close()

def test_responsive_and_cancel_active_worker(service):
    from worker_helpers import paused_cluster_worker, wait_started
    service.supervisor.worker_target = paused_cluster_worker
    ds = service.register_records("local", records=[r.model_dump() for r in tickets()])
    job = service.run_analysis("local", ds["dataset_id"], [{"id":"groups", "type":"cluster"}])
    wait_started(service.config.root, job["job_id"])
    started = time.monotonic()
    assert set(service.get_capabilities("local")["operations"]) == {"cluster", "topics"}
    assert time.monotonic() - started < 1
    assert service.cancel_job("local", job["job_id"])["status"] == "cancelled"
    assert job["job_id"] not in service.supervisor.running

def test_old_source_dataset_rejected_before_enqueue(service):
    ds = service.store.put("local", "ds", {"source_id":"removed", "selection":{}})
    with pytest.raises(VTError, match="register_records"):
        service.run_analysis("local", ds, [{"id":"groups", "type":"cluster"}])
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_completed_cache_reuses_identical_snapshot_with_new_dataset_id(service):
    rows=[r.model_dump() for r in tickets()]
    ops=[{'id':'groups','type':'cluster','parameters':{'eps_values':[.3,.5]}}]
    first_ds=service.register_records('local',records=rows)
    first=wait(service,service.run_analysis('local',first_ds['dataset_id'],ops)['job_id'])
    assert first['status']=='succeeded' and not first['cache_hit']
    second_ds=service.register_records('local',records=rows)
    assert second_ds['dataset_id']!=first_ds['dataset_id']
    cached=service.run_analysis('local',second_ds['dataset_id'],ops,idempotency_key='cached-retry')
    assert cached['status']=='succeeded' and cached['cache_hit']
    assert cached['cached_from_job']==first['job_id']
    assert cached['results']==first['results'] and cached['job_id']!=first['job_id']
    assert cached['job_id'] not in service.supervisor.running
    assert service.run_analysis('local',second_ds['dataset_id'],ops,idempotency_key='cached-retry')['job_id']==cached['job_id']
    with pytest.raises(VTError,match='different request'):
        service.run_analysis('local',first_ds['dataset_id'],ops,idempotency_key='cached-retry')
    forced=service.run_analysis('local',second_ds['dataset_id'],ops,execution={'use_cache':False})
    assert not forced['cache_hit']
    service.cancel_job('local',forced['job_id'])
    # Identical content owned by another caller must not access the first result.
    service.config.callers['other']={}
    other=service.register_records('other',records=rows)
    isolated=service.run_analysis('other',other['dataset_id'],ops)
    assert not isolated['cache_hit']
    service.cancel_job('other',isolated['job_id'])
    # Changing selection scope must not reuse a retrieval-results result.
    full=service.register_records('local',records=rows,selection_scope='complete_selection')
    miss=service.run_analysis('local',full['dataset_id'],ops)
    assert not miss['cache_hit']
    service.cancel_job('local',miss['job_id'])
    # Missing files cannot produce a successful cache response.
    with service.store.connect() as db:
        filename=db.execute('SELECT path FROM objects WHERE id=?',(first['results']['groups'],)).fetchone()[0]
    (service.config.root/'artifacts'/filename).unlink()
    miss=service.run_analysis('local',second_ds['dataset_id'],ops)
    assert not miss['cache_hit']
    service.cancel_job('local',miss['job_id'])


def test_analysis_cache_identity_covers_content_plan_limits_and_engine(service,monkeypatch):
    import copy
    import eigenlens.cache as cache
    ds=service.register_records('local',records=[r.model_dump() for r in tickets()])
    data=service.store.get('local',ds['dataset_id'],'ds')
    ops=[{'id':'groups','type':'cluster'}]
    key=cache.analysis_key(data,ops,{'backend':'auto'},service.config)
    for field,value in [('text','changed text'),('vector',[1.,2.]),('metadata',{'changed':True})]:
        changed=copy.deepcopy(data);changed['records'][0][field]=value
        assert cache.analysis_key(changed,ops,{'backend':'auto'},service.config)!=key
    changed=copy.deepcopy(data);changed['source']={'collection':'different'}
    assert cache.analysis_key(changed,ops,{'backend':'auto'},service.config)!=key
    assert cache.analysis_key(data,[{'id':'groups','type':'cluster','parameters':{'eps_values':[.2]}}],{'backend':'auto'},service.config)!=key
    service.config.numeric_threads=2
    assert cache.analysis_key(data,ops,{'backend':'auto'},service.config)!=key
    service.config.numeric_threads=1
    service.config.limits.max_clusters+=1
    assert cache.analysis_key(data,ops,{'backend':'auto'},service.config)!=key
    service.config.limits.max_clusters-=1
    monkeypatch.setattr(cache,'engine_identity',lambda:{'code':'new implementation'})
    assert cache.analysis_key(data,ops,{'backend':'auto'},service.config)!=key


def test_completed_cache_survives_restart(tmp_path,monkeypatch):
    monkeypatch.setenv('EIGENLENS_DATA',str(tmp_path))
    rows=[r.model_dump() for r in tickets()]
    ops=[{'id':'groups','type':'cluster','parameters':{'eps_values':[.3]}}]
    service=Service()
    try:
        ds=service.register_records('local',records=rows)
        first=wait(service,service.run_analysis('local',ds['dataset_id'],ops)['job_id'])
        assert first['status']=='succeeded'
    finally:service.close()
    restarted=Service()
    try:
        ds=restarted.register_records('local',records=rows)
        cached=restarted.run_analysis('local',ds['dataset_id'],ops)
        assert cached['cache_hit'] and cached['cached_from_job']==first['job_id']
    finally:restarted.close()
