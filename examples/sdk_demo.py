"""Run: python examples/sdk_demo.py. No database or API credentials required."""
import json
import vector_tools as vt
from vector_tools.demo import tickets
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest

def main():
    data = vt.from_records(tickets(), name="synthetic-support-tickets").materialize()
    clusters = vt.cluster(data, estimator=MiniBatchKMeans(n_clusters=4, random_state=42, n_init=3))
    results = {"clusters": clusters, "topics": vt.topics(data, clusters),
               "sentiment": vt.sentiment(data, clusters=clusters),
               "outliers": vt.outliers(data, estimator=IsolationForest(random_state=42))}
    print(json.dumps({name: {"summary": result.summary(), "examples": result.examples()} for name, result in results.items()}, indent=2))

if __name__ == "__main__":
    main()
