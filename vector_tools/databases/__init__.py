"""Compatibility imports; use eigenlens.databases for new code."""
from eigenlens.databases import *
from importlib import import_module as _import_module

def __getattr__(name):
    return getattr(_import_module("eigenlens.databases"), name)
