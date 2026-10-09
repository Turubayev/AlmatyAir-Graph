"""REST API прогноза PM2.5 (FastAPI). Документация: http://localhost:8000/docs

Режимы:
* демо (по умолчанию) — прогноз по архиву data/sample/almaty_demo.csv.gz на любой момент
  тестового периода; можно сравнить прогноз с «фактом»;
* live — /api/forecast/live: свежий прогноз погоды Open-Meteo + PM2.5 из OpenAQ/CAMS.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .interp import interpolate
from .metrics import AQI_BREAKS, AQI_LABELS
from .pipeline import ForecastService, alerts, load_long, prepare
from .stations import stations_frame

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("ALMATYAIR_MODELS", ROOT / "models"))
DATA_PATH = Path(os.environ.get("ALMATYAIR_DATA", ROOT / "data" / "sample" / "almaty_demo.csv.gz"))
COLOC_PATH = Path(os.environ.get("ALMATYAIR_COLOCATION", DATA_PATH.parent / "colocation_demo.csv"))
WEB_DIR = ROOT / "web"

app = FastAPI(title="AlmatyAir-Graph", version=__version__,
              description="Прогноз PM2.5 для Алматы на 1–48 ч с интервалом неопределённости (q10–q90).")


@lru_cache(maxsize=1)
def service() -> ForecastService:
    if not (MODEL_DIR / "meta.json").exists():
        raise HTTPException(503, "Модели не обучены: выполните `python scripts/train.py`")
    return ForecastService(MODEL_DIR)


@lru_cache(maxsize=1)
def demo_data():
    if not DATA_PATH.exists():
        raise HTTPException(503, f"Нет данных {DATA_PATH}: выполните `python scripts/make_synthetic.py`")
    data, _ = prepare(load_long(DATA_PATH), colocation=COLOC_PATH if COLOC_PATH.exists() else None)
    return data


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["valid_time"] = df["valid_time"].astype(str)
    for c in ("q10", "q50", "q90", "observed"):
        if c in df:
            df[c] = df[c].round(1)
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "models": (MODEL_DIR / "meta.json").exists()}


@app.get("/api/stations")
def stations():
    return stations_frame().reset_index().to_dict(orient="records")


@app.get("/api/aqi")
def aqi_scale():
    return {"breaks_ugm3": AQI_BREAKS, "labels": AQI_LABELS, "standard": "US EPA 2024, PM2.5"}


@app.get("/api/demo/range")
def demo_range():
    d = demo_data()
    split = service().meta["split"]
    return {"start": str(d.times[48]), "end": str(d.times[-49]), "test_start": split["val_end"]}


@app.get("/api/forecast")
def forecast(issue_time: str = Query(..., description="Момент выпуска, напр. 2025-01-15T06:00"),
             model: str = Query("auto", pattern="^(auto|stgnn|lightgbm)$"),
             station_id: str | None = None):
    try:
        fc = service().forecast(demo_data(), pd.Timestamp(issue_time).floor("h"), model=model)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    if station_id:
        fc = fc[fc["station_id"] == station_id]
    return {"issue_time": issue_time, "source": "demo", "forecast": _records(fc), "alerts": alerts(fc),
            "quality": {"operational_validation": False, "training_revision": service().meta.get('training_revision',1),
                        "note": "Bundled synthetic/legacy checkpoints must be retrained and evaluated on issued forecasts before operational use."}}


@app.get("/api/forecast/live")
def forecast_live(model: str = Query("auto", pattern="^(auto|stgnn|lightgbm)$")):
    from .live import build_live

    try:
        data, now, source = build_live(stations_frame())
    except Exception as e:  # сеть, лимиты API
        raise HTTPException(502, f"Не удалось получить данные: {e}") from e
    fc = service().forecast(data, now, model=model)
    return {"issue_time": str(now), "source": source, "forecast": _records(fc), "alerts": alerts(fc),
            "quality": {"operational_validation": False, "training_revision": service().meta.get('training_revision',1)}}


@app.get("/api/map")
def pm_map(issue_time: str, lead_h: int = Query(24, ge=1, le=48), model: str = "auto"):
    fc = service().forecast(demo_data(), pd.Timestamp(issue_time).floor("h"), model=model)
    st = stations_frame()
    sub = fc[fc["lead_h"] == lead_h].set_index("station_id").reindex(st.index)
    return {"issue_time": issue_time, "lead_h": lead_h, **interpolate(sub["q50"].to_numpy(), st)}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")
