"""Compatibility module; use eigenlens.databases.weaviate."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.databases.weaviate")
