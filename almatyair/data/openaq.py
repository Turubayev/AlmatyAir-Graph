"""Клиент OpenAQ API v3 (нужен бесплатный ключ: https://explore.openaq.org/register).

Ключ передаётся через переменную окружения OPENAQ_API_KEY.
PM2.5 в OpenAQ — parameter id 2.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2


class OpenAQ:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self.key:
            raise RuntimeError("Не задан OPENAQ_API_KEY (бесплатная регистрация на explore.openaq.org)")
        self.s = requests.Session()
        self.s.headers["X-API-Key"] = self.key

    def _get(self, path: str, **params) -> dict:
        for i in range(4):
            r = self.s.get(f"{BASE}{path}", params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** i * 5)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("OpenAQ: превышен лимит запросов")

    def locations(self, lat: float, lon: float, radius_m: int = 25000) -> pd.DataFrame:
        """Станции с датчиками PM2.5 в радиусе от точки."""
        js = self._get("/locations", coordinates=f"{lat},{lon}", radius=radius_m, parameters_id=PM25_PARAMETER_ID, limit=1000)
        rows = []
        for loc in js["results"]:
            for s in loc.get("sensors", []):
                if s["parameter"]["id"] != PM25_PARAMETER_ID:
                    continue
                rows.append({
                    "location_id": loc["id"],
                    "sensor_id": s["id"],
                    "name": loc.get("name"),
                    "provider": (loc.get("provider") or {}).get("name"),
                    "is_monitor": loc.get("isMonitor"),  # True — эталонная станция
                    "lat": loc["coordinates"]["latitude"],
                    "lon": loc["coordinates"]["longitude"],
                    "first": (loc.get("datetimeFirst") or {}).get("utc"),
                    "last": (loc.get("datetimeLast") or {}).get("utc"),
                })
        return pd.DataFrame(rows)

    def sensor_hours(self, sensor_id: int, date_from: str, date_to: str) -> pd.DataFrame:
        """Часовые средние PM2.5 по датчику (с пагинацией)."""
        rows, page = [], 1
        while True:
            js = self._get(f"/sensors/{sensor_id}/hours", datetime_from=date_from, datetime_to=date_to, limit=1000, page=page)
            res = js.get("results", [])
            for m in res:
                rows.append({"time_utc": m["period"]["datetimeFrom"]["utc"], "pm25": m["value"]})
            if len(res) < 1000:
                break
            page += 1
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time_utc"], utc=True).dt.tz_convert("Asia/Almaty").dt.tz_localize(None)
        return df[["time", "pm25"]]
