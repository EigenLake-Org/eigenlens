"""Shared pure-numpy time helpers for temporal vector operators.

Parsing is intentionally liberal (ISO strings, ``YYYY-MM-DD HH:MM:SS``,
epoch seconds/ms, datetime/date objects) because source payloads vary;
bucket keys are lexicographically sortable ISO labels so chronological
order equals sorted order.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from threading import Lock
from typing import Any

import numpy as np

from .schemas import VTError

TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


def parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            ts = float(value) / 1000.0 if abs(float(value)) > 1e12 else float(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    for fmt in TIME_FORMATS:
        try:
            dt = datetime.strptime(s[: len(fmt)], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError:
        return None


def bucket_key(dt: datetime, bucket: str) -> str:
    if bucket == "day":
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    if bucket == "week":
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if bucket == "month":
        return f"{dt.year:04d}-{dt.month:02d}"
    raise ValueError(f"Unknown bucket {bucket!r}; expected day, week or month")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def centroid(vectors: np.ndarray) -> np.ndarray:
    c = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    n = float(np.linalg.norm(c))
    return c if n == 0.0 else c / n


def cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def cohesion(vectors: np.ndarray, center: np.ndarray) -> float:
    return float(np.mean(vectors @ center))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


BUCKETS = ("day", "week", "month")
_FRAME_CACHE_MAX = 8


@dataclass(frozen=True)
class TemporalFrame:
    """Bucketing computed once and shared by every temporal op on a dataset.

    ``matrix`` holds one row per dated record (in ``usable_positions`` order),
    L2-normalized iff ``normalize``. ``buckets`` maps each *kept* label to the
    matrix-row positions inside it; ``order`` is the kept labels in
    chronological order (sorted order == chronological for ISO labels).
    Small buckets (``dropped``) are excluded from ``buckets``/``order`` but
    their records remain in ``matrix`` so row positions stay stable.
    """

    time_field: str
    bucket: str
    min_bucket_n: int
    normalize: bool
    order: tuple[str, ...]
    buckets: dict[str, tuple[int, ...]]
    usable_positions: tuple[int, ...]
    missing_time: int
    dropped: tuple[str, ...] = ()
    small_bucket_records: int = 0
    matrix: np.ndarray = field(default=None, compare=False, repr=False)

    @property
    def kept_records(self) -> int:
        return sum(len(self.buckets[k]) for k in self.order)

    @classmethod
    def build(cls, data, time_field=None, bucket=None, min_bucket_n=None, normalize=True):
        time_field = time_field or "published_at"
        bucket = bucket or "month"
        min_bucket_n = min_bucket_n or 5
        normalize = True if normalize is None else bool(normalize)
        if bucket not in BUCKETS:
            raise VTError("bucket must be day, week or month")
        if not isinstance(time_field, str) or not time_field:
            raise VTError("time_field must be a nonempty metadata key")
        if not 2 <= min_bucket_n <= 100000:
            raise VTError("min_bucket_n must be 2..100000")
        labels: list[str | None] = []
        missing = 0
        for r in data.records:
            dt = parse_time((r.metadata or {}).get(time_field))
            if dt is None:
                missing += 1
                labels.append(None)
            else:
                labels.append(bucket_key(dt, bucket))
        usable_positions = tuple(i for i, lab in enumerate(labels) if lab is not None)
        if not usable_positions:
            raise VTError(f"No parsable time values in metadata field {time_field!r}")
        if any(data.records[i].vector is None for i in usable_positions):
            raise VTError("Temporal operations require dense vectors; materialize with vector projection")
        x = np.asarray([data.records[i].vector for i in usable_positions], dtype=np.float64)
        if x.nbytes * 8 > data.limits.max_working_bytes:
            raise VTError("Estimated working memory exceeds max_working_bytes")
        matrix = l2_normalize(x) if normalize else x
        ulabels = [labels[i] for i in usable_positions]
        full_order = sorted(set(ulabels))
        all_buckets: dict[str, list[int]] = {k: [] for k in full_order}
        for pos, lab in enumerate(ulabels):
            all_buckets[lab].append(pos)
        kept = tuple(k for k in full_order if len(all_buckets[k]) >= min_bucket_n)
        dropped = tuple(k for k in full_order if len(all_buckets[k]) < min_bucket_n)
        return cls(time_field=time_field, bucket=bucket, min_bucket_n=min_bucket_n,
                   normalize=normalize, order=kept,
                   buckets={k: tuple(all_buckets[k]) for k in kept},
                   usable_positions=usable_positions, missing_time=missing,
                   dropped=dropped,
                   small_bucket_records=sum(len(all_buckets[k]) for k in dropped),
                   matrix=matrix)


_FRAME_CACHE: "OrderedDict[tuple, TemporalFrame]" = OrderedDict()
_FRAME_CACHE_LOCK = Lock()


def get_frame(data, time_field=None, bucket=None, min_bucket_n=None, normalize=True) -> TemporalFrame:
    """Return the cached frame for this dataset + bucketing, building it once.

    The key is content-addressed (``data.fingerprint``), so a shared frame is
    returned for every temporal op in one ``analyze()`` call and across calls
    on the same materialization. Bounded LRU (8 entries): frames hold a full
    embedding matrix each.
    """
    key = (data.fingerprint, time_field or "published_at", bucket or "month",
           min_bucket_n or 5, True if normalize is None else bool(normalize))
    with _FRAME_CACHE_LOCK:
        if key in _FRAME_CACHE:
            _FRAME_CACHE.move_to_end(key)
            return _FRAME_CACHE[key]
    frame = TemporalFrame.build(data, time_field, bucket, min_bucket_n, normalize)
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[key] = frame
        while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
            _FRAME_CACHE.popitem(last=False)
    return frame


def clear_frame_cache() -> None:
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE.clear()
