FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md tune.py ./
COPY aniso_ok_tuner_3d ./aniso_ok_tuner_3d
COPY configs ./configs

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "aniso_ok_tuner_3d.api:app", "--host", "0.0.0.0", "--port", "8000"]
