"""Same SDK workflow for each configured source, including first-class retrieval IDs.
VECTOR_ML_SOURCE_ID=qdrant VECTOR_ML_CONFIG=config/local.json python -m examples.source_sdk
For Pinecone without enumeration, supply VECTOR_ML_IDS='["record-id", ...]'.
"""
import json
import os
import vector_tools as vt
from vector_tools.config import Config, close_source
from examples.mcp_client import OPERATIONS

def main():
    source = Config().source(os.environ["VECTOR_ML_SOURCE_ID"])
    try:
        dataset = vt.Dataset(source)
        ids = json.loads(os.environ["VECTOR_ML_IDS"]) if os.getenv("VECTOR_ML_IDS") else None
        data = dataset.ids(ids).materialize() if ids is not None else dataset.sample(62, seed=42).materialize()
        results = vt.analyze(data, OPERATIONS)
        # Real retrieval-result IDs can now be reused with any connector.
        retrieved = dataset.ids([r.id for r in data.records[:3]]).materialize()
        print(json.dumps({"source":data.source, "scope":data.scope, "retrieval_ids":[r.id for r in retrieved.records],
                          "results":{k:{"summary":v.summary(), "examples":v.examples()} for k,v in results.items()}}, indent=2))
    finally:
        close_source(source)

if __name__ == "__main__":
    main()
