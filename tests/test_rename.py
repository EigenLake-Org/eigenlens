"""Renaming the public API must preserve existing imports and stored settings."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from cryptography.fernet import Fernet
import pytest
import eigenlens as el
import vector_tools as legacy
from eigenlens.config import Config
from eigenlens.connections import ConnectionStore
from eigenlens.environment import get_env


@pytest.fixture
def clean_environment(monkeypatch):
    for name in os.environ:
        if name.startswith(('EIGENLENS_','VECTOR_ML_')):
            monkeypatch.delenv(name)
    return monkeypatch


def test_legacy_imports_share_classes_modules_and_adapter_registry(monkeypatch):
    assert legacy.Record is el.Record and legacy.from_records is el.from_records
    for name in ('schemas','config','service','server','retrieval','databases.base'):
        assert importlib.import_module('vector_tools.'+name) is importlib.import_module('eigenlens.'+name)
    old=importlib.import_module('vector_tools.databases')
    new=importlib.import_module('eigenlens.databases')
    assert old.CUSTOM is new.CUSTOM
    replacement=object()
    monkeypatch.setattr('vector_tools.retrieval.embed_query',replacement)
    assert importlib.import_module('eigenlens.retrieval').embed_query is replacement


def test_old_environment_and_new_precedence(clean_environment,tmp_path):
    env=clean_environment
    env.setenv('VECTOR_ML_DATA',str(tmp_path/'old'))
    env.setenv('VECTOR_ML_WORKERS','2')
    env.setenv('VECTOR_ML_CPU_BUDGET','4')
    config=Config()
    assert config.root==tmp_path/'old' and config.limits.max_jobs==2 and config.numeric_threads==2
    env.setenv('EIGENLENS_DATA',str(tmp_path/'new'))
    env.setenv('EIGENLENS_WORKERS','1')
    assert Config().root==tmp_path/'new' and Config().limits.max_jobs==1
    env.setenv('VECTOR_ML_MCP_TOKEN','legacy-token')
    assert get_env('EIGENLENS_MCP_TOKEN')=='legacy-token'
    env.setenv('EIGENLENS_MCP_TOKEN','new-token')
    assert get_env('VECTOR_ML_MCP_TOKEN')=='new-token'


def test_existing_local_data_is_reused(clean_environment,tmp_path):
    clean_environment.chdir(tmp_path)
    assert Config().root==tmp_path/'.eigenlens'
    (tmp_path/'.vector-ml').mkdir()
    assert Config().root==tmp_path/'.vector-ml'
    (tmp_path/'.eigenlens').mkdir()
    assert Config().root==tmp_path/'.eigenlens'


def test_encrypted_connections_survive_environment_rename(clean_environment,tmp_path):
    key=Fernet.generate_key().decode()
    clean_environment.setenv('VECTOR_ML_CONNECTION_KEY',key)
    store=ConnectionStore(tmp_path)
    spec={'db_type':'qdrant','url':'https://example.invalid','api_key':'test-secret','options':{}}
    saved=store.put('local',spec)
    clean_environment.delenv('VECTOR_ML_CONNECTION_KEY')
    clean_environment.setenv('EIGENLENS_CONNECTION_KEY',key)
    assert ConnectionStore(tmp_path).get('local',saved['connection_id'])==spec


def test_both_plugin_groups_and_new_group_precedence(monkeypatch):
    from types import SimpleNamespace
    from eigenlens.databases import plugins, metadata
    old=SimpleNamespace(name='custom',group='vector_tools.database_adapters')
    new=SimpleNamespace(name='custom',group='eigenlens.database_adapters')
    monkeypatch.setattr(metadata,'entry_points',lambda group:[old] if group==old.group else [new])
    assert plugins()=={'custom':new}


@pytest.mark.parametrize('module',['eigenlens.server','vector_tools.server'])
def test_primary_and_legacy_module_entry_points(module):
    result=subprocess.run([sys.executable,'-m',module,'--help'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and '--transport' in result.stdout
