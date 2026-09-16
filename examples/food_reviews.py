"""Reproducible pre-embedded review demo. No embedding API calls are made.

setup: validate a pinned public CSV and upsert its existing vectors into Qdrant.
analyze: run real ML through MCP and write a readable report, without an LLM.
agent: use the optional OpenAI Agents SDK with a live model on the same source.
"""
from __future__ import annotations
import argparse
import ast
import asyncio
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "config" / "food_reviews.lock.json"


def read_lock(path=DEFAULT_LOCK):
    return json.loads(Path(path).read_text())


def download(lock, cache_dir):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "food_reviews.csv"
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == lock["sha256"]:
        return target
    temporary = cache / ("download-" + uuid.uuid4().hex + ".tmp")
    try:
        print("Downloading the pinned public dataset (about 35 MB); no API key required.", flush=True)
        with urllib.request.urlopen(lock["url"], timeout=60) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 40 * 1024 * 1024:
                    raise ValueError("Dataset download exceeds the expected size limit")
                output.write(chunk)
        data = temporary.read_bytes()
        if len(data) != lock["bytes"] or hashlib.sha256(data).hexdigest() != lock["sha256"]:
            raise ValueError("Dataset checksum/size mismatch; refusing to index changed content")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def parse_csv(data: bytes, lock: dict):
    """Parse literal numeric vectors without eval; validate every record before any write."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    required = {"", "ProductId", "Score", "Summary", "Text", "combined", "embedding"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Dataset CSV is missing required columns")
    records = []
    ids = set()
    for row in reader:
        original_id = row[""]
        if not original_id or original_id in ids:
            raise ValueError("Dataset review IDs must be nonempty and unique")
        ids.add(original_id)
        vector = ast.literal_eval(row["embedding"])
        if not isinstance(vector, list) or len(vector) != lock["dimensions"]:
            raise ValueError(f"Review {original_id}: invalid embedding dimension")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise ValueError(f"Review {original_id}: embedding must contain finite numbers")
        if not any(vector):
            raise ValueError(f"Review {original_id}: zero vector")
        score = int(row["Score"])
        if score not in range(1, 6) or not row["combined"].strip():
            raise ValueError(f"Review {original_id}: missing text or invalid star rating")
        id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"openai-food-reviews/{lock['sha256']}/{original_id}"))
        records.append({
            "id": id, "vector": vector,
            "payload": {
                "text": row["combined"], "summary": row["Summary"], "score": score,
                "product_id": row["ProductId"], "review_id": original_id, "parent_id": original_id,
                "dataset": "Amazon Fine Food Reviews / OpenAI Cookbook 1k sample",
                "dataset_sha256": lock["sha256"], "embedding_provider": "OpenAI",
                "embedding_model": lock["embedding_model"], "source_url": lock["url"],
            },
        })
    if len(records) != lock["expected_rows"]:
        raise ValueError(f"Expected {lock['expected_rows']} records, found {len(records)}")
    return records


def setup(args):
    from qdrant_client import QdrantClient, models
    lock = read_lock(args.lock)
    path = download(lock, args.cache_dir)
    records = parse_csv(path.read_bytes(), lock)
    expected_ids = {r["id"] for r in records}
    client = QdrantClient(url=args.qdrant_url, timeout=60)
    try:
        if client.collection_exists(args.collection):
            vectors = client.get_collection(args.collection).config.params.vectors
            cfg = vectors.get("body") if isinstance(vectors, dict) else None
            if cfg is None or cfg.size != lock["dimensions"] or cfg.distance != models.Distance.COSINE:
                raise ValueError("Existing collection has incompatible vector configuration; choose another collection")
            offset = None
            while True:
                points, offset = client.scroll(args.collection, limit=256, offset=offset,
                    with_vectors=False, with_payload=["dataset_sha256"])
                if any(str(p.id) not in expected_ids or (p.payload or {}).get("dataset_sha256") != lock["sha256"] for p in points):
                    raise ValueError("Existing collection contains other data; refusing to mix datasets. Choose another collection.")
                if offset is None:
                    break
        else:
            client.create_collection(args.collection, vectors_config={
                "body": models.VectorParams(size=lock["dimensions"], distance=models.Distance.COSINE)
            })
        for start in range(0, len(records), 64):
            batch = records[start:start + 64]
            client.upsert(args.collection, points=[models.PointStruct(
                id=r["id"], vector={"body": r["vector"]}, payload=r["payload"]
            ) for r in batch], wait=True)
            print(f"Indexed {min(start + 64, len(records))}/{len(records)} reviews", flush=True)
        actual = client.count(args.collection, exact=True).count
        if actual != len(records):
            raise ValueError(f"Post-load count mismatch: {actual}")
        # Verify ID, text, rating and vector alignment against actual stored records.
        for start in range(0, len(records), 64):
            batch = records[start:start + 64]
            found = {str(p.id): p for p in client.retrieve(args.collection, ids=[r["id"] for r in batch], with_vectors=["body"], with_payload=True)}
            for row in batch:
                point = found.get(row["id"])
                if point is None or point.payload["text"] != row["payload"]["text"] or point.payload["score"] != row["payload"]["score"]:
                    raise ValueError("Indexed ID/text/rating alignment check failed")
                original, stored = row["vector"], point.vector["body"]
                norm = math.sqrt(sum(v * v for v in original))
                if len(stored) != len(original) or any(abs(a / norm - b) > 1e-5 for a, b in zip(original, stored)):
                    raise ValueError("Stored vector differs from the expected cosine-normalized input")
        manifest = {**lock, "collection": args.collection, "records": actual,
            "rating_counts": dict(sorted(Counter(str(r["payload"]["score"]) for r in records).items())),
            "verification": "Every ID, text, rating and normalized stored vector checked",
            "embedding_api_calls": 0}
        (Path(args.cache_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Ready: {actual} reviews in {args.collection}; existing embeddings reused.")
    finally:
        client.close()


def operations(clusters=8):
    return [
        {"id": "groups", "type": "cluster", "parameters": {"n_clusters": clusters, "random_state": 42}},
        {"id": "themes", "type": "topics", "clusters_from": "groups"},
        {"id": "sentiment", "type": "sentiment", "clusters_from": "groups"},
        {"id": "outliers", "type": "outliers"},
    ]


def markdown_report(report):
    summaries = report["summaries"]
    groups = summaries["groups"]
    sentiment = summaries["sentiment"]
    lines = ["# Food Reviews Demo", "", "Computed locally through the MCP service; no LLM or embedding API calls.", "",
        f"- Records analyzed: **{groups['processed']}**", f"- Selection scope: `{groups['scope']}`",
        f"- Job: `{report['job']['job_id']}`", f"- Dataset fingerprint: `{groups['dataset_fingerprint']}`",
        f"- Sentiment counts: {sentiment['summary_data']['labels']}", "",
        "## Discovered groups", "", "| Group | Reviews | Top terms | Mean VADER sentiment |", "|---|---:|---|---:|"]
    topics = {str(t["cluster_id"]): t["terms"] for t in summaries["themes"]["summary_data"]["topics"]}
    sizes = groups["summary_data"]["cluster_sizes"]
    for cid, count in sorted(sizes.items()):
        terms = ", ".join(t["term"] for t in topics.get(cid, [])[:6])
        mean = sentiment["summary_data"]["by_cluster"].get(cid, {}).get("mean_compound")
        lines.append(f"| {cid} | {count} | {terms} | {mean:.3f} |" if mean is not None else f"| {cid} | {count} | {terms} | — |")
    lines += ["", "## Representative reviews", ""]
    for cid, result in report["cluster_examples"].items():
        for row in result["examples"]:
            # Quotes are escaped as code-free block text, not interpreted as Markdown/HTML.
            text = row["text"].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
            lines += [f"### Group {cid} · review {row['parent_id']} · `{row['id']}`", "", "> " + text, ""]
    lines += ["## Unusual reviews", "", "Higher public anomaly scores mean more unusual embedding patterns; these are not confirmed errors or fraud.", ""]
    for row in report["outlier_examples"]["examples"]:
        text = row["text"].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
        lines += [f"- Review **{row['parent_id']}**, anomaly score **{row['anomaly_score']:.4f}**: {text}"]
    lines += ["", "## Interpretation", "",
        "This is a public 1,000-review sample, not the entire Amazon review population. Counts are review records. "
        "VADER is an English lexicon/rule baseline; scores are not calibrated probabilities. "
        "The CSV supplies OpenAI embeddings but does not declare their exact model identity. "
        "No embeddings were regenerated. Star ratings are comparison signals, not sentiment ground truth.", ""]
    if report.get("rating_comparison"):
        lines += ["## Star-rating comparison", "", "| Stars | Scored reviews | Mean VADER compound |", "|---|---:|---:|"]
        for stars, info in sorted(report["rating_comparison"].items()):
            lines.append(f"| {stars} | {info['count']} | {info['mean_compound']:.3f} |")
        lines += ["", "Ratings were joined to scored records by the dataset's deterministic point IDs."]
    return "\n".join(lines) + "\n"


async def analyze(args):
    from examples.mcp_client import connection, content, poll
    trace = []
    async with connection("http") as (session, initialized):
        async def call(name, arguments):
            result = content(await session.call_tool(name, arguments))
            trace.append({"tool": name, "arguments": arguments, "result": result})
            return result
        await call("get_capabilities", {})
        description = await call("describe_source", {"source_id": args.source})
        dataset = await call("create_dataset", {"source_id": args.source, "selection": {}})
        submitted = await call("run_analysis", {"dataset_id": dataset["dataset_id"], "operations": operations(args.clusters), "idempotency_key": uuid.uuid4().hex})
        print(f"Submitted {submitted['job_id']}; polling without model calls.", flush=True)
        job = await poll(call, submitted["job_id"], timeout=300)
        if job["status"] != "succeeded":
            raise RuntimeError(f"Analysis {job['status']}: {job['error']}")
        summaries = {name: await call("get_summary", {"result_id": ref}) for name, ref in job["results"].items()}
        cluster_examples = {cid: await call("get_examples", {"result_id": job["results"]["groups"], "cluster_id": int(cid), "limit": 1}) for cid in summaries["groups"]["summary_data"]["cluster_sizes"]}
        outliers = await call("get_examples", {"result_id": job["results"]["outliers"], "limit": 5})
        report = {"mode": "Real MCP and ML; no model stub and no LLM calls", "source": description,
            "dataset": dataset, "job": job, "summaries": summaries, "cluster_examples": cluster_examples,
            "outlier_examples": outliers, "protocol_version": initialized.protocolVersion}
        path = Path(args.cache_dir) / "food_reviews.csv"
        if path.exists():
            lock = read_lock(args.lock)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != lock["sha256"]:
                raise ValueError("Cached dataset checksum mismatch")
            ratings = {r["id"]: r["payload"]["score"] for r in parse_csv(raw, lock)}
            grouped = {}
            cursor = None
            while True:
                page = await call("get_result_page", {"result_id": job["results"]["sentiment"], "cursor": cursor, "limit": 200})
                for row in page["rows"]:
                    if row["id"] not in ratings:
                        raise ValueError("Analysis IDs do not match the pinned review dataset")
                    grouped.setdefault(str(ratings[row["id"]]), []).append(row["scores"]["compound"])
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            report["rating_comparison"] = {stars: {"count": len(values), "mean_compound": sum(values) / len(values)} for stars, values in grouped.items()}
        report["trace"] = trace
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "food-review-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "food-review-report.md").write_text(markdown_report(report))
    print(f"Completed: {summaries['groups']['processed']} reviews. Report: {output / 'food-review-report.md'}")


async def agent(args):
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"):
        raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL for the live agent. Use analyze for the no-key demo.")
    from examples.agent_client import connection, workflow
    async with connection("http") as server:
        report = await workflow(server, os.environ["OPENAI_MODEL"], args.source,
            sample_n=None, n_clusters=args.clusters, subject="food reviews")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "food-review-agent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report.get("final_answer") or report)
    if report["status"] != "completed":
        raise RuntimeError(f"Agent workflow ended with status {report['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "analyze", "agent"])
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="food_reviews")
    parser.add_argument("--source", default="food_reviews")
    parser.add_argument("--cache-dir", default="work/food-cache")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--output", default="demo-output")
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.clusters <= 256:
        parser.error("clusters must be 1..256")
    os.environ.setdefault("VECTOR_ML_MCP_URL", "http://localhost:8000/mcp")
    os.environ.setdefault("VECTOR_ML_MCP_TOKEN", "local-demo-token")
    try:
        if args.command == "setup":
            setup(args)
        else:
            asyncio.run(analyze(args) if args.command == "analyze" else agent(args))
    except Exception as exc:
        print(f"Food review demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
