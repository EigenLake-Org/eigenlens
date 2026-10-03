"""BANKING77 preparation, SDK analysis, MCP and optional OpenAI agent entry points."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import uuid
import vector_tools as vt
from vector_tools.config import Config

OPERATIONS = [{"id": "groups", "type": "cluster", "algorithm": "dbscan_search",
               "parameters": {"pca_components": 50, "label_field": "label_text", "random_state": 42}}]
ARTIFACT = "banking77.jsonl"


def prepare(args, config):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi
    # Resolve mutable refs to immutable commits before downloading and record them.
    api = HfApi()
    dataset_revision = api.dataset_info("mteb/banking77", revision=args.dataset_revision).sha
    model_revision = api.model_info("sentence-transformers/all-MiniLM-L6-v2", revision=args.model_revision).sha
    ds = load_dataset("mteb/banking77", revision=dataset_revision)
    rows = []
    label_feature = ds["train"].features.get("label")
    names = getattr(label_feature, "names", None)
    for split in ("train", "test"):
        for i, row in enumerate(ds[split]):
            label = row.get("label_text")
            if label is None:
                label = names[int(row["label"])] if names else str(row["label"])
            rows.append({"id": f"banking77:{split}:{i}", "text": row["text"],
                         "metadata": {"label": row["label"], "label_text": label, "split": split}})
    if args.limit and len(rows) > args.limit:
        import numpy as np
        selected = sorted(np.random.default_rng(42).choice(len(rows), args.limit, replace=False))
        rows = [rows[i] for i in selected]
    if len(rows) > config.limits.max_rows:
        raise ValueError("Dataset exceeds configured max_rows")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=model_revision)
    vectors = model.encode([r["text"] for r in rows], batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    target = config.root / "ingress" / "local" / ARTIFACT
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    with temp.open('w') as f:
        for row, vector in zip(rows, vectors):
            row["vector"] = vector.tolist()
            f.write(vt.Record.model_validate(row).model_dump_json() + '\n')
    if temp.stat().st_size > config.limits.max_bytes:
        temp.unlink()
        raise ValueError("Prepared artifact exceeds configured max_bytes")
    temp.replace(target)
    manifest = {"dataset": "mteb/banking77", "dataset_revision": dataset_revision,
                "model": "sentence-transformers/all-MiniLM-L6-v2", "model_revision": model_revision,
                "rows": len(rows), "dimensions": int(vectors.shape[1]), "seed": 42,
                "selection": "seeded_sample" if args.limit else "train_plus_test",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def read_records(config):
    path = config.root / "ingress" / "local" / ARTIFACT
    if not path.exists():
        raise SystemExit("Run python -m examples.banking77 prepare first")
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Prepared artifact checksum mismatch; prepare again")
    return [vt.Record.model_validate_json(line) for line in path.read_text().splitlines()], manifest


def export(result, records, output, manifest):
    import pandas as pd
    import plotly.express as px
    import html
    output.mkdir(parents=True, exist_ok=True)
    summary = result.summary_data
    (output / "banking77_result.json").write_text(result.model_dump_json(indent=2))
    (output / "banking77_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(summary["parameter_search"]).to_csv(output / "banking77_dbscan_parameter_search.csv", index=False)
    groups = pd.DataFrame(summary["groups"])
    groups.to_csv(output / "banking77_cluster_summary.csv", index=False)
    lookup = {r.id: r for r in records}
    terms = {g["cluster_id"]: ', '.join(g["top_terms"]) for g in summary["groups"]}
    table = []
    for assignment in result.rows:
        r = lookup[assignment["id"]]
        cid = assignment["cluster_id"]
        table.append(dict(id=r.id, query=r.text, true_intent=r.metadata.get("label_text"),
                          cluster_id=cid, terms=terms[cid], PC1=assignment["pca_3d"][0],
                          PC2=assignment["pca_3d"][1], PC3=assignment["pca_3d"][2],
                          cluster_name=("Unassigned" if summary["status"] != "selected" else "Noise") if cid == -1 else f"Cluster {cid} | {terms[cid]}"))
    frame = pd.DataFrame(table)
    frame.to_csv(output / "banking77_clustered.csv", index=False)
    pd.crosstab(frame.cluster_id, frame.true_intent).to_csv(output / "banking77_cluster_vs_intent.csv")
    display = frame.copy()
    for column in ("query", "true_intent", "terms", "cluster_name"):
        display[column] = display[column].map(lambda v: html.escape(str(v)))
    fig = px.scatter_3d(display, x="PC1", y="PC2", z="PC3", color="cluster_name", hover_name="query",
                        hover_data=["true_intent", "terms"], title=html.escape(f"BANKING77 — DBSCAN ({summary['status']})" + (f" · {summary['query']}" if summary.get("query") else "")))
    fig.update_traces(marker=dict(size=2, opacity=.65))
    fig.write_html(output / "banking77_dbscan_3d.html", include_plotlyjs=True)
    print(f"Reports saved to {output.resolve()}; status={summary['status']}")


async def remote(args, config, records, manifest):
    from examples.mcp_client import content, poll
    from examples.agent_client import connection, workflow
    from vector_tools.schemas import Result
    # This example owns a local stdio server so staged files share the filesystem.
    async with connection("stdio") as server:
        async def call(name, arguments):
            return content(await server.call_tool(name, arguments))
        ds = await call("register_records", {"artifact_name": ARTIFACT})
        if args.command == "agent":
            if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_MODEL"):
                raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL for the live agent")
            run = await workflow(server, os.environ["OPENAI_MODEL"], dataset_id=ds["dataset_id"],
                                 operations=OPERATIONS, subject="BANKING77 intents; report search status, noise, silhouette, ARI/NMI and limitations",
                                 poll_timeout=config.limits.max_seconds + 60)
            print(run["final_answer"])
            jobs = run["jobs"]
            if not jobs:
                raise RuntimeError("Agent did not submit analysis")
            job = jobs[-1]
        else:
            job = await call("run_analysis", {"dataset_id": ds["dataset_id"], "operations": OPERATIONS,
                                               "idempotency_key": uuid.uuid4().hex})
            job = await poll(call, job["job_id"], timeout=config.limits.max_seconds + 60)
        if job["status"] != "succeeded":
            raise RuntimeError(job)
        ref = job["results"]["groups"]
        summary = await call("get_summary", {"result_id": ref})
        rows, cursor = [], None
        while True:
            page = await call("get_result_page", {"result_id": ref, "limit": 1000, **({"cursor": cursor} if cursor else {})})
            rows.extend(page["rows"])
            cursor = page["next_cursor"]
            if not cursor:
                break
        export(Result(**summary, rows=rows), records, Path(args.output), manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "analyze", "mcp", "agent"])
    parser.add_argument("--limit", type=int, default=0, help="0 loads all rows; positive values use a seeded sample")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output", default="demo-output/banking77")
    parser.add_argument("--config", default="config/banking77.json")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    os.environ["VECTOR_ML_CONFIG"] = str(Path(args.config).resolve())
    config = Config()
    if args.command == "prepare":
        prepare(args, config)
        return
    records, manifest = read_records(config)
    if args.command == "analyze":
        data = vt.from_records(records, name="banking77", limits=config.limits).materialize()
        if manifest["selection"] == "seeded_sample":
            data.scope = "sample"
        export(vt.analyze(data, OPERATIONS)["groups"], records, Path(args.output), manifest)
    else:
        asyncio.run(remote(args, config, records, manifest))


if __name__ == "__main__":
    main()
