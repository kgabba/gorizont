# ГОРИЗОНТ (gorizo.ru)

Веб-приложение для 3D-анизотропии и подбора параметров Ordinary Kriging.

## Архитектура

```text
Frontend (Next.js)
  → FastAPI (/api/analyze)
    → anisotropy3d        (MOI + variogram ellipsoid)
    → aniso_candidate gate
    → kriging3d_tuner     (Optuna search neighbourhood)
```

Математика в `backend/anisotropy3d` и `backend/kriging3d_tuner` не изменяется.
Оркестрация — `backend/pipeline3d`. HTTP-слой — `backend/api`.

## CSV

Обязательные колонки (регистр важен):

- `X`, `Y`, `Z`, `Value`

Опционально: `HoleID`.

Минимум точек задаётся в конфигах модулей (`n_min_points`, по умолчанию 30).

## API

Через nginx: `https://gorizo.ru/api/...` → `gorizo-api:8000`.

| Method | Path | Описание |
|--------|------|----------|
| GET | `/api/health` | `{"status":"ok"}` |
| POST | `/api/analyze` | multipart: `file` (CSV), optional `trials` |

Результаты прогона сохраняются в `runs/<run_id>/`:

```text
runs/<run_id>/
  input.csv
  anisotropy/
    result.json
    *.png / *.csv
  kriging_tuning/
    best_params.json
    trials.csv
  pipeline_summary.json
```

## Локальный запуск backend

```bash
cd backend
pip install -e ./anisotropy3d
pip install -e ./kriging3d_tuner
pip install -e .
pip install fastapi uvicorn python-multipart
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

CLI пайплайна (без API):

```bash
cd backend
python -m pipeline3d \
  --data data.csv \
  --aniso-config anisotropy3d/config.yaml \
  --tuner-config kriging3d_tuner/config.yaml \
  --out-dir runs/test01 \
  --trials 50
```

## Production / Docker

Из корня репозитория (сеть `georisk-pro_pro` должна существовать):

```bash
docker compose up -d --build
```

Сервисы:

- `gorizo-web` — Next.js
- `gorizo-api` — FastAPI + математические пакеты

Каталог `runs/` монтируется в контейнер API и не коммитится в git.
