# syntax=docker/dockerfile:1
FROM python:3.13-slim AS server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECTOR_ML_DATA=/data
WORKDIR /app
COPY pyproject.toml requirements-tested.txt ./
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; print('\n'.join(p['dependencies'] + sum([p['optional-dependencies'][e] for e in ['server','sentiment','qdrant','weaviate','pgvector','pinecone']], [])))" > /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -c requirements-tested.txt -r /tmp/requirements.txt && useradd --uid 10001 --create-home vector && mkdir /data && chown vector:vector /data
COPY README.md ./
COPY vector_tools ./vector_tools
RUN pip install --no-deps .
COPY config /app/config
USER vector
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["vector-ml", "--transport", "http", "--host", "0.0.0.0", "--config", "/app/config/docker.json"]


# Optional demo client: the MCP server image itself remains independent of Agents.
FROM server AS demo_client
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install -c requirements-tested.txt 'openai-agents>=0.22.2,<0.23' && mkdir -p /cache /output && chown vector:vector /cache /output
COPY examples /app/examples
USER vector
HEALTHCHECK NONE
CMD ["python", "-m", "examples.food_reviews", "analyze", "--cache-dir", "/cache", "--output", "/output"]

# Query-job server: owns embedding, Qdrant retrieval, clustering and plot generation.
FROM server AS query_jobs
USER root
RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install '.[banking]' && mkdir -p /models && chown vector:vector /models
COPY examples /app/examples
ENV HF_HOME=/models TOKENIZERS_PARALLELISM=false
USER vector
EXPOSE 8010
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)"
CMD ["python", "-m", "examples.query_job_server", "--host", "0.0.0.0", "--config", "/app/config/query-jobs-docker.json"]

# Plain docker build defaults to the server, while Compose explicitly selects demo_client.
FROM server AS runtime
