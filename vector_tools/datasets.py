"""Compatibility module; use eigenlens.datasets."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.datasets")
