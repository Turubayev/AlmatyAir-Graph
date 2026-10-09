"""Калибровка недорогих оптических датчиков PM2.5 по эталонным станциям.

Оптические датчики завышают PM2.5 при высокой влажности: частицы набухают.
Используем модель гигроскопического роста (упрощённая κ-Кёлер):

    pm_raw ≈ a + b · pm_true · (1 + κ · RH² / (1 − RH))

Коэффициенты a, b, κ подбираются по совместным измерениям «датчик ↔ станция»
(данные одной точки или соседних точек в радиусе ~2 км). Обратное преобразование
даёт исправленную концентрацию.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def _growth(rh_pct: np.ndarray, kappa: float) -> np.ndarray:
    r = np.clip(np.asarray(rh_pct, float) / 100.0, 0.0, 0.97)
    return 1.0 + kappa * r**2 / (1.0 - r)


@dataclass
class HumidityCalibration:
    a: float = 0.0
    b: float = 1.0
    kappa: float = 0.0

    def fit(self, raw, rh, ref) -> "HumidityCalibration":
        raw, rh, ref = map(lambda x: np.asarray(x, float), (raw, rh, ref))
        ok = np.isfinite(raw) & np.isfinite(rh) & np.isfinite(ref) & (ref > 0)
        raw, rh, ref = raw[ok], rh[ok], ref[ok]
        if len(raw) < 24:
            raise ValueError("Слишком мало совместных измерений для калибровки (нужно ≥ 24 ч)")

        def resid(p):
            a, b, k = p
            # Ошибка в лог-пространстве: одинаково важны чистые и грязные часы.
            return np.log1p(np.clip(a + b * ref * _growth(rh, k), 0, None)) - np.log1p(raw)

        res = least_squares(resid, x0=[0.0, 1.0, 0.2], bounds=([-20, 0.2, 0.0], [20, 5.0, 3.0]))
        self.a, self.b, self.kappa = map(float, res.x)
        return self

    def correct(self, raw, rh) -> np.ndarray:
        raw = np.asarray(raw, float)
        out = (raw - self.a) / (self.b * _growth(rh, self.kappa))
        return np.clip(out, 0, None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HumidityCalibration":
        return cls(**json.loads(Path(path).read_text()))


def calibrate_network(df: pd.DataFrame, stations: pd.DataFrame, colocation: pd.DataFrame | None = None,
                      max_km: float = 5.0, max_dh: float = 100.0) -> tuple[pd.DataFrame, dict]:
    """Исправляет показания lowcost-датчиков; результат — колонка `pm25`.

    Рекомендуемый способ — `colocation`: таблица совместных измерений (pm25_raw, rh, pm25_ref),
    снятая при размещении датчика того же типа рядом со станцией Казгидромета. Полученные
    коэффициенты применяются ко всем датчикам сети.

    Без co-location используется ближайшая официальная станция, но только если она ближе
    `max_km` и на той же высоте (± `max_dh` м): иначе в коэффициенты попадёт реальная разница
    загрязнения между районами, и калибровка «сотрёт» её.
    """
    from .stations import haversine_km

    df = df.copy()
    df["pm25"] = df["pm25_raw"]
    params: dict[str, dict] = {}
    lowcost = stations.index[stations["kind"] == "lowcost"]
    if colocation is not None:
        cal = HumidityCalibration().fit(colocation["pm25_raw"], colocation["rh"], colocation["pm25_ref"])
        m = df["station_id"].isin(lowcost)
        df.loc[m, "pm25"] = cal.correct(df.loc[m, "pm25_raw"], df.loc[m, "rh"])
        for sid in lowcost:
            params[sid] = {**asdict(cal), "reference": "co-location", "distance_km": 0.0}
        return df, params

    wide_ref = df.pivot(index="time", columns="station_id", values="pm25_ref")
    official = stations.index[stations["kind"] == "official"]
    for sid in lowcost:
        cands = []
        for o in official:
            d = haversine_km(stations.at[sid, "lat"], stations.at[sid, "lon"], stations.at[o, "lat"], stations.at[o, "lon"])
            dh = abs(stations.at[sid, "elevation"] - stations.at[o, "elevation"])
            if d <= max_km and dh <= max_dh:
                cands.append((d, o))
        if not cands:
            continue  # датчик остаётся некалиброванным, это видно по params
        dist, ref_id = min(cands)
        sub = df[df["station_id"] == sid].set_index("time")
        cal = HumidityCalibration().fit(sub["pm25_raw"], sub["rh"], wide_ref[ref_id].reindex(sub.index))
        m = df["station_id"] == sid
        df.loc[m, "pm25"] = cal.correct(df.loc[m, "pm25_raw"], df.loc[m, "rh"])
        params[sid] = {**asdict(cal), "reference": ref_id, "distance_km": round(float(dist), 1)}
    return df, params
