"""Сеть станций мониторинга PM2.5 в Алматы.

Координаты и высоты — ориентировочные (центры районов). Перед работой с реальными
данными замените их фактическими координатами станций из OpenAQ / Казгидромета:
`python scripts/fetch_openaq.py --list` выводит станции в радиусе 25 км.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

ALMATY_CENTER = (43.238, 76.889)


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    lat: float
    lon: float
    elevation: float  # м над уровнем моря
    kind: str  # "official" (Казгидромет) | "lowcost" (оптический датчик)
    private_sector: float  # доля частного сектора с угольным отоплением вокруг, 0..1
    tpp_distance_km: float  # расстояние до ТЭЦ-2


STATIONS: list[Station] = [
    Station("ALT", "Алатауский р-н", 43.296, 76.820, 680, "lowcost", 0.75, 6.0),
    Station("TRK", "Турксибский р-н", 43.335, 76.950, 670, "official", 0.60, 11.0),
    Station("ZHT", "Жетысуский р-н", 43.285, 76.930, 720, "lowcost", 0.45, 9.0),
    Station("ALM", "Алмалинский р-н", 43.255, 76.915, 790, "official", 0.10, 9.5),
    Station("AUE", "Ауэзовский р-н", 43.230, 76.840, 820, "lowcost", 0.25, 7.5),
    Station("NAU", "Наурызбайский р-н", 43.190, 76.790, 950, "lowcost", 0.65, 11.5),
    Station("BOS", "Бостандыкский р-н", 43.215, 76.910, 880, "official", 0.15, 11.0),
    Station("MED", "Медеуский р-н", 43.230, 76.965, 900, "lowcost", 0.20, 13.0),
    Station("KAM", "Каменское плато", 43.170, 76.940, 1250, "official", 0.05, 15.0),
]

STATION_IDS = [s.id for s in STATIONS]


def stations_frame(stations: list[Station] | None = None) -> pd.DataFrame:
    """Таблица станций. Переменная ALMATYAIR_STATIONS=путь.csv подменяет встроенный список."""
    path = os.environ.get("ALMATYAIR_STATIONS")
    if stations is None and path:
        return pd.read_csv(path).set_index("id")
    return pd.DataFrame([asdict(s) for s in (stations or STATIONS)]).set_index("id")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Азимут от точки 1 к точке 2, градусы (0 = север, по часовой)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360
