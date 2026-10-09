"""Граф станций с учётом расстояния, рельефа и ветра.

Статический вес ребра i→j:
    w_ij = exp(−d²/σ_d²) · exp(−|Δh| / σ_h)
Город лежит на склоне (перепад высот — сотни метров), поэтому станции на одной
высоте связаны сильнее, чем станции на одном расстоянии, но на разных «этажах» инверсии.

Динамический (ветровой) множитель: воздух переносится по ветру,
    w_ij(t) = w_ij · (1 + α · max(0, cos(θ_wind_to − азимут j→i)))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stations import bearing_deg, haversine_km


def distance_matrix(st: pd.DataFrame) -> np.ndarray:
    lat, lon = st["lat"].to_numpy(), st["lon"].to_numpy()
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def static_adjacency(st: pd.DataFrame, sigma_km: float = 8.0, sigma_h: float = 250.0) -> np.ndarray:
    d = distance_matrix(st)
    h = st["elevation"].to_numpy()
    dh = np.abs(h[:, None] - h[None, :])
    w = np.exp(-(d**2) / sigma_km**2) * np.exp(-dh / sigma_h)
    np.fill_diagonal(w, 0.0)
    return w


def wind_adjacency(st: pd.DataFrame, wind_dir_deg: float, alpha: float = 1.5, **kw) -> np.ndarray:
    """Вес w[i, j] — влияние станции j на станцию i при заданном направлении ветра (откуда дует)."""
    w = static_adjacency(st, **kw)
    lat, lon = st["lat"].to_numpy(), st["lon"].to_numpy()
    brg_j_to_i = bearing_deg(lat[None, :], lon[None, :], lat[:, None], lon[:, None])
    to_dir = (wind_dir_deg + 180) % 360
    return w * (1 + alpha * np.clip(np.cos(np.radians(brg_j_to_i - to_dir)), 0, None))


def normalize(a: np.ndarray, self_loops: bool = True) -> np.ndarray:
    """Симметричная нормировка D^-1/2 (A + I) D^-1/2 (как в GCN)."""
    if self_loops:
        a = a + np.eye(len(a))
    d = a.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.where(d > 0, d, 1))
    return dinv[:, None] * a * dinv[None, :]


def neighbour_mean(values: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Взвешенное среднее соседей; values: (..., N), пропуски (NaN) игнорируются."""
    v = np.nan_to_num(values, nan=0.0)
    present = np.isfinite(values).astype(float)
    num = v @ adj.T
    den = present @ adj.T
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
