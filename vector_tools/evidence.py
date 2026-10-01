"""LLM-ready evidence packs rendered from operation results.

Format ``evidence/1.0``: a short prose-plus-IDs rendering of ``Result`` objects,
designed as retrieval output for generation. Every renderer asserts
non-degeneracy and raises ``VTError`` with an actionable message instead of
emitting a pack that would mislead a model (e.g. novelty without record text,
all-zero shares).
"""
from __future__ import annotations

from collections.abc import Mapping

from .schemas import Result, VTError

EVIDENCE_VERSION = "1.0"


def _require(condition, message):
    if not condition:
        raise VTError(message)


def _header(result: Result) -> str:
    params = result.effective_parameters
    scope = params.get("bucket") or params.get("representation") or result.operation
    return (f"[{result.operation} {result.operation_id} scope={scope} "
            f"usable={result.summary_data.get('usable_records', result.processed)}]")


def render_drift(result: Result) -> list[str]:
    rows = result.rows
    _require(len(rows) >= 2, "drift evidence needs at least two buckets; "
                             "check min_bucket_n and time coverage")
    for r in rows:
        _require(isinstance(r.get("drift_prev"), (int, float)) or r.get("drift_prev") is None,
                 f"drift evidence: non-numeric drift_prev in bucket {r.get('bucket')}")
        _require(isinstance(r.get("cohesion"), (int, float)),
                 f"drift evidence: non-numeric cohesion in bucket {r.get('bucket')}")
    lines = [_header(result) + " drift=1-cosine centroid shift; burst=volume z>1.5 (*=edge bucket)"]
    for r in rows:
        mark = "*" if r.get("edge_bucket") else ""
        lines.append(f"- {r['bucket']}{mark}: n={r['n']} cohesion={r['cohesion']} "
                     f"drift_prev={r['drift_prev']} burst={r['burst']}")
    return lines


def render_novelty(result: Result, top_n: int = 12) -> list[str]:
    rows = result.rows
    _require(rows, "novelty evidence needs probe records; "
                    "check min_bucket_n and time coverage")
    cal = result.summary_data
    text_of = {e.get("id"): (e.get("text") or "").strip() for e in result.example_rows}
    top = rows[:max(1, top_n)]
    missing = [r["id"] for r in top if not text_of.get(r["id"])]
    if missing:
        raise VTError("novelty evidence needs record text for top records "
                      f"({len(missing)} missing, e.g. {missing[0]}); "
                      "materialize with text projection")
    lines = [_header(result) +
             f" novelty=1-max cosine to prior buckets (p50={cal.get('novelty_p50')} p90={cal.get('novelty_p90')})"]
    for r in top:
        lines.append(f"- nov={r['novelty']} {r['bucket']} :: {text_of[r['id']][:220]}")
    return lines


def render_flow(result: Result, terms: dict | None = None) -> list[str]:
    rows = result.rows
    _require(len(rows) >= 2, "flow evidence needs at least two buckets; "
                             "check min_bucket_n and time coverage")
    shares = [r.get("largest_share") for r in rows]
    _require(any(isinstance(s, (int, float)) and s > 0 for s in shares),
             "flow evidence is degenerate (no positive largest_share); "
             "refusing to render")
    lines = [_header(result) + " JSD base-2 between consecutive cluster histograms"]
    for r in rows:
        extra = f" terms={terms.get(str(r['largest_cluster']))}" if terms and str(r.get("largest_cluster")) in terms else ""
        lines.append(f"- {r['bucket']}: n={r['n']} largest=c{r['largest_cluster']} "
                     f"share={r['largest_share']} jsd_prev={r['jsd_vs_prev']}{extra}")
    return lines


def render_topics(result: Result, top_n: int = 8) -> dict:
    """Topics render to a cluster->terms lookup shared with flow, not lines."""
    if result.summary_data.get("vocabulary_status") == "empty_vocabulary":
        return {}
    terms = {}
    for r in result.rows:
        terms[str(r["cluster_id"])] = [t["term"] for t in r.get("terms", [])[:top_n]]
    return terms


def render_result(result: Result, **kwargs) -> list[str]:
    renderer = {"drift": render_drift, "novelty": render_novelty, "flow": render_flow}.get(result.operation)
    if renderer is None:
        params = result.effective_parameters
        return [_header(result) + f" params={params} summary_keys={sorted(result.summary_data)}"]
    if result.operation == "flow" and kwargs.get("terms") is not None:
        return renderer(result, terms=kwargs["terms"])
    if result.operation == "novelty" and kwargs.get("top_n") is not None:
        return renderer(result, top_n=kwargs["top_n"])
    return renderer(result)


def render_evidence(results: Mapping[str, Result], max_chars: int = 12000) -> str:
    """Render a combined evidence pack for several results of one dataset.

    ``topics`` results contribute a cluster->terms lookup (not lines) so
    ``flow`` lines carry human-readable terms. Raises ``VTError`` if any
    renderer finds degenerate output or if results span datasets.
    """
    results = dict(results)
    _require(results, "render_evidence needs at least one result")
    fps = {r.dataset_fingerprint for r in results.values()}
    _require(len(fps) == 1, "render_evidence needs results from one dataset; "
                            f"got {len(fps)} fingerprints")
    terms: dict = {}
    for r in results.values():
        if r.operation == "topics":
            terms.update(render_topics(r))
    lines = [f"# Evidence (vector-tools evidence/{EVIDENCE_VERSION}; one dataset, {len(results)} ops)"]
    for op_id, r in results.items():
        lines.extend(render_result(r, terms=terms) if r.operation == "flow" else render_result(r))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return text
