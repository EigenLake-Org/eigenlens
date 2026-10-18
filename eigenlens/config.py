"""Worker limits, storage and authenticated caller configuration."""
import json
import os
from .environment import get_env
import re
from pathlib import Path
from .schemas import Limits, VTError

class Config:
    def __init__(self, path=None):
        self.path = path or get_env("EIGENLENS_CONFIG")
        self.raw = json.loads(Path(self.path).read_text()) if self.path else {"callers": {"local": {}}}
        limits=dict(self.raw.get("limits", {}))
        workers=get_env("EIGENLENS_WORKERS")
        if workers is not None:
            if not re.fullmatch(r"([1-9]|1[0-6])",workers):
                raise VTError("EIGENLENS_WORKERS must be an integer from 1 to 16")
            limits["max_jobs"]=int(workers)
        total=get_env("EIGENLENS_TOTAL_WORKING_BYTES")
        if total is not None:
            if not total.isascii() or not total.isdigit():
                raise VTError("EIGENLENS_TOTAL_WORKING_BYTES must be a positive byte count")
            count=limits.get("max_jobs",Limits().max_jobs)
            limits["max_working_bytes"]=int(total)//count
        self.limits = Limits.model_validate(limits)
        budget = get_env("EIGENLENS_CPU_BUDGET", "1")
        if not re.fullmatch(r"(?:[1-9]|[1-5][0-9]|6[0-4])", budget):
            raise VTError("EIGENLENS_CPU_BUDGET must be an integer from 1 to 64")
        self.numeric_threads = max(1, int(budget) // self.limits.max_jobs)
        directory = get_env("EIGENLENS_DATA", self.raw.get("data_dir", ".eigenlens"))
        self.root = Path(directory).resolve()
        self.callers = self.raw.get("callers", {"local": {}})
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", owner) for owner in self.callers):
            raise VTError("Caller IDs must use letters, digits, underscore or hyphen")
