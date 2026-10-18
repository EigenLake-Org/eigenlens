"""EigenLens configuration with aliases for existing deployments."""
import os


def get_env(name, default=None):
    for prefix in ('EIGENLENS_', 'VECTOR_ML_'):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            return os.environ.get('EIGENLENS_' + suffix,
                                  os.environ.get('VECTOR_ML_' + suffix, default))
    return os.environ.get(name, default)
