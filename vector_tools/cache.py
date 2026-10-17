"""Content-based, caller-scoped cache identity for completed analysis jobs."""
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

from .datasets import canonical


@lru_cache(maxsize=1)
def engine_identity():
    root = Path(__file__).parent
    code = hashlib.sha256()
    for name in ('cache.py', 'density.py', 'operators.py', 'backends.py', 'registry.py',
                 'datasets.py', 'records.py', 'schemas.py', 'execution_context.py', 'jobs.py'):
        code.update(name.encode())
        code.update((root / name).read_bytes())
    return {'code': code.hexdigest(), 'python': platform.python_version(),
            'platform': [platform.system(), platform.machine()],
            'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'bertopic')}}


def analysis_key(dataset, operations, execution, config):
    return hashlib.sha256(canonical({'dataset': dataset, 'operations': operations,
        'execution': {k:v for k,v in execution.items() if k!='use_cache'}, 'limits': config.limits.model_dump(),
        'numeric_threads': config.numeric_threads, 'engine': engine_identity()}).encode()).hexdigest()
