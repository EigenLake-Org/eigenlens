"""Compatibility module; use eigenlens.server."""
import sys
from importlib import import_module

if __name__ == "__main__":
    import_module("eigenlens.server").main()
else:
    sys.modules[__name__] = import_module("eigenlens.server")
