"""Клиент Open-Meteo (бесплатно, без ключа для некоммерческого использования).

* Архив ERA5 / ERA5-Land:      https://archive-api.open-meteo.com/v1/archive
* Архив прошлых прогнозов:     https://historical-forecast-api.open-meteo.com/v1/forecast
* Оперативный прогноз (GFS/ICON …): https://api.open-meteo.com/v1/forecast
* Качество воздуха CAMS:        https://air-quality-api.open-meteo.com/v1/air-quality

CAMS используется как обязательная базовая модель для сравнения (см. README).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]
RENAME = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "boundary_layer_height": "blh",
}
TZ = "Asia/Almaty"


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429 and i < retries - 1:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Open-Meteo: превышено число попыток")


def _hourly_frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns=RENAME)


def weather_history(lat: float, lon: float, start: str, end: str, source: str = "era5") -> pd.DataFrame:
    """Почасовая метеорология. source='era5' — реанализ, 'forecast' — архив прогнозов на сутки вперёд."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY),
        "wind_speed_unit": "ms",
        "timezone": TZ,
    }
    url = ARCHIVE_URL if source == "era5" else HIST_FC_URL
    if source == "era5":
        params["models"] = "era5"
    return _hourly_frame(_get(url, params))


def weather_forecast(lat: float, lon: float, hours: int = 72, model: str | None = None) -> pd.DataFrame:
    """Оперативный прогноз погоды на `hours` часов вперёд (+ прошедшие сутки для истории)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY),
        "wind_speed_unit": "ms",
        "timezone": TZ,
        "past_hours": 48,
        "forecast_hours": hours,
    }
    if model:
        params["models"] = model  # например "gfs_seamless" или "icon_seamless"
    return _hourly_frame(_get(FORECAST_URL, params))


def cams_pm25(lat: float, lon: float, start: str | None = None, end: str | None = None, forecast_days: int = 3) -> pd.DataFrame:
    """PM2.5 из CAMS (анализ + прогноз). Возвращает колонки time, pm25_cams."""
    params = {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "timezone": TZ}
    if start and end:
        params.update(start_date=start, end_date=end)
    else:
        params.update(past_days=2, forecast_days=forecast_days)
    df = _hourly_frame(_get(AQ_URL, params))
    return df.rename(columns={"pm2_5": "pm25_cams"})
