"""Compatibility module; use eigenlens.service."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.service")
