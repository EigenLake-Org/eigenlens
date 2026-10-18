"""Compatibility module; use eigenlens.execution_context."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.execution_context")
