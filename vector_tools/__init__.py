"""Compatibility imports; use eigenlens for new code."""
from eigenlens import *
from importlib import import_module as _import_module

def __getattr__(name):
    return getattr(_import_module("eigenlens"), name)
