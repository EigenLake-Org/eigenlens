"""Compatibility module; use eigenlens.density."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.density")
