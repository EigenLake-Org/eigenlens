"""Compatibility module; use eigenlens.artifacts."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.artifacts")
