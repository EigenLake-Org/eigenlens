"""Worker limits, storage and authenticated caller configuration."""
import json
import os
import re
from pathlib import Path
from .schemas import Limits, VTError

class Config:
    def __init__(self, path=None):
        self.path = path or os.getenv("VECTOR_ML_CONFIG")
        self.raw = json.loads(Path(self.path).read_text()) if self.path else {"callers": {"local": {}}}
        limits=dict(self.raw.get("limits", {}))
        workers=os.getenv("VECTOR_ML_WORKERS")
        if workers is not None:
            if not re.fullmatch(r"([1-9]|1[0-6])",workers):
                raise VTError("VECTOR_ML_WORKERS must be an integer from 1 to 16")
            limits["max_jobs"]=int(workers)
        total=os.getenv("VECTOR_ML_TOTAL_WORKING_BYTES")
        if total is not None:
            if not total.isascii() or not total.isdigit():
                raise VTError("VECTOR_ML_TOTAL_WORKING_BYTES must be a positive byte count")
            count=limits.get("max_jobs",Limits().max_jobs)
            limits["max_working_bytes"]=int(total)//count
        self.limits = Limits.model_validate(limits)
        self.root = Path(os.getenv("VECTOR_ML_DATA", self.raw.get("data_dir", ".vector-ml"))).resolve()
        self.callers = self.raw.get("callers", {"local": {}})
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", owner) for owner in self.callers):
            raise VTError("Caller IDs must use letters, digits, underscore or hyphen")
