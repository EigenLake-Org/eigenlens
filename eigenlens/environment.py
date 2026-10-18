"""Environment settings for EigenLens."""
import os

def get_env(name, default=None):
    return os.environ.get(name, default)
