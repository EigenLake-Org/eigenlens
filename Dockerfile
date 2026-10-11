# syntax=docker/dockerfile:1
FROM python:3.13-slim AS server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECTOR_ML_DATA=/data
WORKDIR /app
COPY pyproject.toml requirements-tested.txt ./
RUN python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['server']))" > /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -c requirements-tested.txt -r /tmp/requirements.txt && apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --create-home vector && mkdir /data && chown vector:vector /data
COPY README.md ./
COPY vector_tools ./vector_tools
RUN pip install --no-deps .
COPY config /app/config
USER vector
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2)"
CMD ["vector-ml", "--transport", "http", "--host", "0.0.0.0", "--config", "/app/config/chat-worker.json"]
