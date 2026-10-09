"""Карта PM2.5: интерполяция между станциями с учётом рельефа.

Вес станции для точки сетки: w = 1/d² · exp(−|Δh| / σ_h). Высота точки берётся из
цифровой модели рельефа, если она передана (например, Copernicus DEM GLO-30, массив
той же формы, что и сетка), иначе — из линейной аппроксимации высот станций
(город поднимается с севера на юг к горам).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stations import haversine_km

BBOX = (43.15, 76.74, 43.36, 77.02)  # юг, запад, север, восток


def elevation_trend(st: pd.DataFrame):
    A = np.c_[np.ones(len(st)), st["lat"], st["lon"]]
    coef, *_ = np.linalg.lstsq(A, st["elevation"].to_numpy(), rcond=None)
    return lambda lat, lon: coef[0] + coef[1] * lat + coef[2] * lon


def grid(bbox=BBOX, n_lat: int = 40, n_lon: int = 50):
    lats = np.linspace(bbox[0], bbox[2], n_lat)
    lons = np.linspace(bbox[1], bbox[3], n_lon)
    return np.meshgrid(lats, lons, indexing="ij")


def interpolate(values: np.ndarray, st: pd.DataFrame, bbox=BBOX, n_lat: int = 40, n_lon: int = 50,
                dem: np.ndarray | None = None, sigma_h: float = 200.0, power: float = 2.0) -> dict:
    """values — PM2.5 по станциям (в порядке st). Возвращает сетку и значения."""
    LAT, LON = grid(bbox, n_lat, n_lon)
    elev = dem if dem is not None else elevation_trend(st)(LAT, LON)
    ok = np.isfinite(values)
    s = st[ok]
    v = np.asarray(values)[ok]
    d = haversine_km(LAT[..., None], LON[..., None], s["lat"].to_numpy(), s["lon"].to_numpy())
    dh = np.abs(elev[..., None] - s["elevation"].to_numpy())
    w = 1.0 / np.maximum(d, 0.3) ** power * np.exp(-dh / sigma_h)
    z = (w * v).sum(-1) / w.sum(-1)
    return {"lat": LAT[:, 0].tolist(), "lon": LON[0].tolist(), "pm25": np.round(z, 1).tolist(), "bbox": list(bbox)}
