FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

# Kernel from zip — copied as-is, not modified
COPY kriging3d_tuner/ ./

RUN pip install --no-cache-dir -e .

# No HTTP API in this kernel; keep container alive for CLI use inside.
CMD ["python", "-c", "import kriging3d_tuner; print('kriging3d_tuner ready'); import time; time.sleep(86400 * 365)"]
