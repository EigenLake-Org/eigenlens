"""Index MiniLM BANKING77 vectors, query Qdrant, cluster hits and export a plot."""
from contextlib import closing
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from qdrant_client import QdrantClient, models
import vector_tools as vt
from vector_tools.config import Config
from examples.banking77 import prepare, read_records, export

DEFAULT_QUERY = "Banking customer support problems with cards, payments, transfers, cash withdrawals and accounts"


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


def retrieve(client, collection, query_vector, manifest, top_k):
    if not 3 <= top_k <= 2000:
        raise ValueError('top_k must be 3..2000 for bounded clustering')
    response = client.query_points(collection_name=collection, using='body', query=query_vector,
                                   query_filter=dataset_filter(manifest), limit=top_k,
                                   with_payload=True, with_vectors=['body'])
    records, hits = [], []
    for rank, point in enumerate(response.points, 1):
        payload = dict(point.payload or {})
        if payload.get('embedding_model') != manifest['model'] or payload.get('embedding_revision') != manifest['model_revision']:
            raise ValueError('Query and indexed embedding model revisions do not match')
        records.append(vt.Record(id=str(point.id), vector=point.vector['body'], text=payload.pop('text'), metadata=payload))
        hits.append({'rank':rank, 'id':str(point.id), 'score':float(point.score), 'original_id':payload.get('original_id')})
    if len(records) < 3:
        raise ValueError('Query returned fewer than three records; cannot cluster')
    return records, hits


def query_and_plot(client, collection, manifest, query, top_k, output, limits, encoder=None):
    if not query.strip():
        raise ValueError('Query must not be empty')
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(manifest['model'], revision=manifest['model_revision'])
    vector = np.asarray(encoder.encode(query, normalize_embeddings=True, convert_to_numpy=True))
    if vector.shape != (manifest['dimensions'],) or not np.all(np.isfinite(vector)):
        raise ValueError('Query embedding dimensions or values are invalid')
    records, hits = retrieve(client, collection, vector.tolist(), manifest, top_k)
    data = vt.from_records(records, name=collection, limits=limits).materialize()
    data.scope = 'retrieval_results'
    data.source = {'kind':'qdrant', 'collection':collection, 'vector_field':'body'}
    data.selection = {'query':query, 'top_k':top_k, 'returned':len(records), 'metric':'cosine', 'embedding_model':manifest['model'], 'embedding_revision':manifest['model_revision']}
    # A broader bounded grid than the full-corpus example accommodates a query subset.
    operation = {'id':'groups', 'type':'cluster', 'algorithm':'dbscan_search', 'parameters':{
        'pca_components':50, 'eps_values':[round(.3+.05*i,2) for i in range(14)],
        'min_samples_values':[5,10,15], 'label_field':'label_text', 'random_state':42}}
    result = vt.analyze(data, [operation])['groups']
    result.summary_data['query'] = query
    result.summary_data['retrieved_count'] = len(records)
    export(result, records, output, manifest)
    report = {'query':query, 'collection':collection, 'top_k':top_k, 'retrieved':len(records),
              'embedding_model':manifest['model'], 'embedding_revision':manifest['model_revision'],
              'status':result.summary_data['status'], 'best_parameters':result.effective_parameters['best_parameters'],
              'groups':result.summary_data['groups'], 'noise_count':result.summary_data['noise_count'],
              'hits':hits, 'plot':str((output/'banking77_dbscan_3d.html').resolve())}
    (output/'query_results.json').write_text(json.dumps(report, indent=2))
    lines = ['# BANKING77: Qdrant query clusters', '', f"Query: **{query}**", '',
             f"Indexed corpus: {manifest['rows']:,} banking queries. Retrieved: **{len(records):,}**. Scope: retrieval results.",
             f"Model: `{manifest['model']}`. Noise/unassigned: **{report['noise_count']}**.",
             f"Search status: `{report['status']}`.", '', '| Cluster | Queries | Top terms |', '|---|---:|---|']
    for g in report['groups']:
        lines.append(f"| {'Noise' if g['cluster_id']==-1 else g['cluster_id']} | {g['queries']} | {', '.join(g['top_terms'])} |")
    lines += ['', '## Nearest query matches', '']
    for hit, r in list(zip(hits, records))[:10]:
        lines.append(f"- **{hit['score']:.3f}** — {(r.text or '').replace(chr(10), ' ')}")
    lines += ['', 'Cosine scores rank similarity; they are not probabilities. These clusters describe only the retrieved subset. Known intents are evaluation metadata and are not used to select clusters.', '', '[Open interactive 3D plot](banking77_dbscan_3d.html)']
    (output/'query_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'hits','groups'}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup','query','demo'])
    parser.add_argument('--query', default=DEFAULT_QUERY)
    parser.add_argument('--top-k', type=int, default=2000)
    parser.add_argument('--qdrant-url', default='http://localhost:6333')
    parser.add_argument('--collection', default='banking77_minilm')
    parser.add_argument('--config', default='config/banking77.json')
    parser.add_argument('--output', default='demo-output/banking77-query')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--model-revision', default='main')
    args = parser.parse_args()
    config = Config(args.config)
    artifact = config.root/'ingress'/'local'/'banking77.jsonl'
    if not artifact.exists() and args.command in {'setup','demo'}:
        args.limit = 0
        prepare(args, config)
    records, manifest = read_records(config)
    if manifest['selection'] != 'train_plus_test' and args.command in {'setup','demo'}:
        raise SystemExit('Prepared data is a sample. Run python -m examples.banking77 prepare for the full corpus first.')
    with closing(QdrantClient(url=args.qdrant_url, api_key=os.getenv('QDRANT_API_KEY'), timeout=120)) as client:
        if args.command in {'setup','demo'}:
            index_records(client, args.collection, records, manifest)
        if args.command in {'query','demo'}:
            query_and_plot(client, args.collection, manifest, args.query, args.top_k, Path(args.output), config.limits)


if __name__ == '__main__':
    main()
