"""Run: python examples/sdk_demo.py. Synthetic vectors; no credentials required."""
import json
import vector_tools as vt
from vector_tools.demo import tickets


def main():
    data = vt.from_records(tickets(), name="synthetic-support-tickets").materialize()
    results = vt.analyze(data, [
        {"id": "groups", "type": "cluster", "algorithm":"dbscan_search"},
        {"id": "themes", "type": "topics", "clusters_from": "groups", "parameters": {"ngram_max": 2}},
    ])
    print(json.dumps({name: {"summary": result.summary(), "examples": result.examples()}
                      for name, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
