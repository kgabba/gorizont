FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# Math packages from gorizont3d — installed as-is, not modified
COPY anisotropy3d/ ./anisotropy3d/
COPY kriging3d_tuner/ ./kriging3d_tuner/
COPY pipeline3d/ ./pipeline3d/
COPY pyproject.toml ./
COPY api/ ./api/

RUN pip install --no-cache-dir \
  -e ./anisotropy3d \
  -e ./kriging3d_tuner \
  -e . \
  fastapi uvicorn python-multipart

ENV RUNS_DIR=/app/runs
RUN mkdir -p /app/runs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
