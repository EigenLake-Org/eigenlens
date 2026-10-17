"""Per-job numerical thread budget; synchronous SDK calls default to one."""
from contextlib import contextmanager
from contextvars import ContextVar

_THREADS = ContextVar('vector_tools_numeric_threads', default=1)


def numeric_threads():
    return _THREADS.get()


@contextmanager
def numeric_budget(threads):
    token = _THREADS.set(threads)
    try:
        yield
    finally:
        _THREADS.reset(token)
