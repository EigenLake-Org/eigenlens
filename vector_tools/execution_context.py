"""Per-job numerical thread budget; synchronous SDK calls default to one."""
from contextlib import contextmanager
from contextvars import ContextVar

_THREADS = ContextVar('vector_tools_numeric_threads', default=1)
_REPORTER = ContextVar('vector_tools_progress_reporter', default=None)
_OPERATION = ContextVar('vector_tools_progress_operation', default={})


def report_stage(stage, message, **details):
    """Report real work boundaries; synchronous SDK execution needs no reporter."""
    reporter = _REPORTER.get()
    if reporter is not None:
        reporter({"stage": stage, "message": message, **_OPERATION.get(), **details})


@contextmanager
def progress_reporter(reporter):
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


@contextmanager
def operation_progress(operation_id, operation_index, operation_count):
    token = _OPERATION.set(dict(operation_id=operation_id, operation_index=operation_index,
                                operation_count=operation_count))
    try:
        yield
    finally:
        _OPERATION.reset(token)


def numeric_threads():
    return _THREADS.get()


@contextmanager
def numeric_budget(threads):
    token = _THREADS.set(threads)
    try:
        yield
    finally:
        _THREADS.reset(token)
