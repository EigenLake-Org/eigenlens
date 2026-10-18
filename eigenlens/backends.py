"""Execute validated clustering plans in the current process."""
import uuid
from .registry import plan, REGISTRY
from . import operators
from .execution_context import operation_progress, report_stage

class LocalBackend:
    def capabilities(self):
        return {"name": "local", "implemented": True, "operations": list(REGISTRY),
                "execution": "calling process for SDK; dedicated child process for service"}
    def execute(self, data, operations, run_id=None):
        ordered = plan(operations, limits=data.limits)
        if not data.records:
            from .schemas import VTError
            raise VTError("Empty selection: no records to analyze")
        run_id = run_id or "run_" + uuid.uuid4().hex
        results = {}
        for index, op in enumerate(ordered, 1):
            kwargs = {"op_id": op.id, "run_id": run_id, **op.parameters.supplied()}
            if op.type == "cluster":
                kwargs["algorithm"] = op.algorithm or "dbscan_search"
            if op.representation:
                kwargs["representation"] = op.representation
            if op.clusters_from:
                kwargs["clusters"] = results[op.clusters_from]
            with operation_progress(op.id, index, len(ordered)):
                report_stage('starting_operation', f'Starting {op.type} operation')
                results[op.id] = getattr(operators, op.type)(data, **kwargs)
        return results
