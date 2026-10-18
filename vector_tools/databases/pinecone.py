"""Compatibility module; use eigenlens.databases.pinecone."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("eigenlens.databases.pinecone")
