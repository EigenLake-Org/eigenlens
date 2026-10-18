"""Compatibility module; use eigenlens.client_source."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.client_source")
