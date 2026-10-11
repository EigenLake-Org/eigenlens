"""Versioned, JSON-serializable contracts. No estimator or transport classes."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
import math

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"

class VTError(ValueError):
    """Actionable validation, capability, or availability error."""

class Record(Contract):
    id: str
    vector: list[float] | None = Field(default=None, max_length=65536)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    @field_validator("vector")
    @classmethod
    def finite_dense(cls, v):
        if v is not None and (not v or any(not math.isfinite(x) for x in v)):
            raise ValueError("Expected a nonempty finite dense vector")
        return v

class Predicate(Contract):
    field: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "exists"] = "eq"
    value: str | int | float | bool | list[str | int | float | bool] | None = None

class Selection(Contract):
    ids: list[str] | None = None
    filters: list[Predicate] = Field(default_factory=list)
    sample_n: int | None = Field(default=None, ge=1)
    seed: int = 42

class Limits(Contract):
    max_rows: int = Field(default=10000, ge=1)
    max_scan_rows: int = Field(default=100000, ge=1)
    max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    batch_size: int = Field(default=256, ge=1, le=1000)
    max_jobs: int = Field(default=1, ge=1, le=16)
    max_pending_jobs: int = Field(default=32, ge=1)
    max_clusters: int = Field(default=256, ge=1)
    max_working_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_seconds: int = Field(default=300, ge=1)

class Parameters(Contract):
    random_state: int | None = None
    top_n: int | None = Field(default=None, ge=1, le=100)
    ngram_max: int | None = Field(default=None, ge=1, le=3)
    stop_words: Literal["english"] | None = None
    pca_components: int | None = Field(default=None, ge=1, le=512)
    eps_values: list[float] | None = Field(default=None, min_length=1, max_length=32)
    min_samples_values: list[int] | None = Field(default=None, min_length=1, max_length=16)
    max_noise_ratio: float | None = Field(default=None, gt=0, le=1)
    silhouette_sample_size: int | None = Field(default=None, ge=3, le=4000)
    label_field: str | None = Field(default=None, min_length=1, max_length=128)
    @field_validator("eps_values")
    @classmethod
    def positive_eps(cls, values):
        if values is not None and any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("eps_values must be finite and positive")
        return values
    @field_validator("min_samples_values")
    @classmethod
    def positive_samples(cls, values):
        if values is not None and any(v < 1 for v in values):
            raise ValueError("min_samples_values must be positive")
        return values
    def supplied(self):
        return self.model_dump(exclude_none=True, exclude={"schema_version"})

class Operation(Contract):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    type: Literal["cluster", "topics"]
    algorithm: str | None = None
    parameters: Parameters = Field(default_factory=Parameters)
    clusters_from: str | None = None
    representation: Literal["bertopic"] | None = None

class Execution(Contract):
    mode: Literal["async", "sync"] = "async"
    backend: Literal["auto", "local"] = "auto"

class Result(Contract):
    result_id: str
    run_id: str
    operation_id: str
    operation: str
    dataset_fingerprint: str
    scope: str
    source: dict[str, Any]
    processed: int
    skipped: dict[str, int]
    effective_parameters: dict[str, Any]
    versions: dict[str, str]
    created_at: str
    summary_data: dict[str, Any]
    rows: list[dict[str, Any]]
    example_rows: list[dict[str, Any]] = Field(default_factory=list)
    backend: str = "local"
    count_unit: str = "vector_records"
    aggregation_policy: str = "No parent-ticket deduplication; every vector record counts once."
    def summary(self):
        return self.model_dump(exclude={"rows", "example_rows"})
    def examples(self, cluster_id: int | None = None, n: int = 5):
        if not 1 <= n <= 100:
            raise VTError("Example limit must be 1..100")
        return [r for r in self.example_rows if cluster_id is None or r.get("cluster_id") == cluster_id][:n]
    def top(self, n: int = 20):
        if not 1 <= n <= 1000:
            raise VTError("Page size must be 1..1000")
        return self.rows[:n]
