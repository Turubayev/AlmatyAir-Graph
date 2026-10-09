"""Оперативный режим: история PM2.5 + свежий прогноз погоды → WideData для ForecastService.

Источник PM2.5:
* OpenAQ (если задан OPENAQ_API_KEY и в stations.csv указаны sensor_id);
* иначе — CAMS через Open-Meteo как приближение (в ответе API помечается source=cams_proxy).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import openmeteo
from .features import WideData


def build_live(stations: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[WideData, pd.Timestamp, str]:
    now = (now or pd.Timestamp.now(tz="Asia/Almaty").tz_localize(None)).floor("h")
    frames, source = [], "cams_proxy"
    use_openaq = bool(os.environ.get("OPENAQ_API_KEY")) and "sensor_id" in stations
    oaq = None
    if use_openaq:
        from .data.openaq import OpenAQ

        oaq = OpenAQ()
        source = "openaq"
    for sid, s in stations.iterrows():
        met = openmeteo.weather_forecast(s["lat"], s["lon"], hours=60)
        if oaq is not None and pd.notna(s.get("sensor_id")):
            pm = oaq.sensor_hours(int(s["sensor_id"]), str((now - pd.Timedelta(hours=72)).date()), str(now.date() + pd.Timedelta(days=1)))
        else:
            cams = openmeteo.cams_pm25(s["lat"], s["lon"])
            pm = cams.rename(columns={"pm25_cams": "pm25"})
        df = met.merge(pm[["time", "pm25"]], on="time", how="left")
        df.loc[df["time"] > now, "pm25"] = np.nan  # будущее не наблюдаем
        df["station_id"] = sid
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    return WideData.from_long(long, stations), now, source
