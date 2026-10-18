"""Compatibility module; use eigenlens.demo."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.demo")
