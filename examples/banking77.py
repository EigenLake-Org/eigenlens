"""Prepare BANKING77 MiniLM vectors and index them for Vector Chat."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import uuid
import vector_tools as vt
from vector_tools.config import Config
from qdrant_client import QdrantClient, models

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


def point_id(record_id, manifest):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, manifest['sha256'] + ':' + record_id))


def dataset_filter(manifest):
    return models.Filter(must=[models.FieldCondition(key="dataset_sha256", match=models.MatchValue(value=manifest['sha256']))])


def index_records(client, collection, records, manifest):
    """Idempotent upserts isolated by dataset fingerprint; never delete existing data."""
    dimensions = manifest['dimensions']
    if client.collection_exists(collection):
        vectors = client.get_collection(collection).config.params.vectors
        if not isinstance(vectors, dict) or 'body' not in vectors or vectors['body'].size != dimensions or vectors['body'].distance != models.Distance.COSINE:
            raise ValueError("Existing collection has incompatible vectors; use a different collection name")
        total = client.count(collection, exact=True).count
        ours = client.count(collection, count_filter=dataset_filter(manifest), exact=True).count
        if total != ours:
            raise ValueError("Collection contains another dataset/model snapshot; use a new collection name")
    else:
        client.create_collection(collection, vectors_config={'body': models.VectorParams(size=dimensions, distance=models.Distance.COSINE)})
    for start in range(0, len(records), 128):
        points = [models.PointStruct(id=point_id(r.id, manifest), vector={'body':r.vector}, payload={
            **r.metadata, 'text':r.text, 'original_id':r.id, 'dataset_sha256':manifest['sha256'],
            'embedding_model':manifest['model'], 'embedding_revision':manifest['model_revision']})
            for r in records[start:start+128]]
        client.upsert(collection, points=points, wait=True)
    count = client.count(collection, count_filter=dataset_filter(manifest), exact=True).count
    if count != len(records):
        raise ValueError(f"Expected {len(records)} points, got {count}")
    print(f"Indexed and counted {count} MiniLM vectors in Qdrant collection {collection}", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "index", "setup"])
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="banking77_minilm")
    parser.add_argument("--config", default="config/banking77.json")
    args = parser.parse_args()
    args.limit = 0  # Always keep the complete train + test corpus.
    config = Config(args.config)
    path = config.root / "ingress" / "local" / ARTIFACT
    if args.command == "prepare" or (args.command == "setup" and not path.exists()):
        prepare(args, config)
    if args.command == "prepare":
        return
    records, manifest = read_records(config)
    if manifest["selection"] != "train_plus_test":
        raise SystemExit("Cached data is a sample; run prepare to build the full corpus.")
    with closing(QdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY"), timeout=120)) as client:
        index_records(client, args.collection, records, manifest)
    print(f"In Vector Chat, connect to {args.qdrant_url} and choose {args.collection}.")


if __name__ == "__main__":
    main()
