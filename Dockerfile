FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY almatyair ./almatyair
COPY scripts ./scripts
COPY web ./web
COPY data/sample ./data/sample
COPY models ./models

# Если моделей нет в образе — обучаем на демо-данных при сборке (≈ 5–10 мин).
RUN test -f models/meta.json || python scripts/train.py --epochs 10

EXPOSE 8000
CMD ["uvicorn", "almatyair.api:app", "--host", "0.0.0.0", "--port", "8000"]
