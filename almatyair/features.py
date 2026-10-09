"""Подготовка признаков: «широкие» массивы (время × станция) и табличные выборки.

Ключевая идея проекта: на вход подаётся не только история PM2.5, но и прогноз погоды
на горизонт прогноза. Для обучения на истории используется подход «perfect prognosis
+ шум»: метео на момент t+h берётся из реанализа/наблюдений и зашумляется тем сильнее,
чем больше заблаговременность (так модель не переоценивает надёжность метеопрогноза).
Если доступен архив реальных прогнозов (Open-Meteo Historical Forecast API), его можно
подать через `WideData.forecast_archive = ForecastArchive(frame)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .graph import neighbour_mean, static_adjacency
from .stations import stations_frame

HORIZONS = (12, 24, 36, 48)
HIST_HOURS = 48
MAX_H = 48
LAGS = (0, 1, 3, 6, 12, 24)


@dataclass
class WideData:
    times: pd.DatetimeIndex
    station_ids: list[str]
    pm: np.ndarray  # (T, N)
    met: dict[str, np.ndarray]  # temp, rh, wind_speed, wind_dir, blh: (T, N)
    stations: pd.DataFrame
    truth: np.ndarray | None = None  # (T, N), только для синтетики
    adj: np.ndarray = field(init=False)
    forecast_archive: object | None = None

    def __post_init__(self):
        self.pm = np.array(self.pm, dtype=float, copy=True)
        self.met = {k: np.array(v, dtype=float, copy=True) for k,v in self.met.items()}
        if self.pm.shape != (len(self.times),len(self.station_ids)) or any(v.shape != self.pm.shape for v in self.met.values()):
            raise ValueError('Time/station arrays must share a grid')
        self.adj = static_adjacency(self.stations.loc[self.station_ids])

    @classmethod
    def from_long(cls, df: pd.DataFrame, stations: pd.DataFrame | None = None, pm_col: str = "pm25") -> "WideData":
        st = stations if stations is not None else stations_frame()
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        ids = [s for s in st.index if s in set(df["station_id"])]
        times = pd.date_range(df["time"].min(), df["time"].max(), freq="h")

        def wide(col):
            return df.pivot_table(index="time", columns="station_id", values=col, aggfunc="mean").reindex(index=times, columns=ids)

        met = {c: wide(c).ffill(limit=6).to_numpy() for c in ["temp", "rh", "wind_speed", "blh"]}
        # Направление ветра интерполируем через компоненты.
        wd = np.radians(wide("wind_dir"))
        u = np.sin(wd).ffill(limit=6)
        v = np.cos(wd).ffill(limit=6)
        met["wind_dir"] = np.mod(np.degrees(np.arctan2(u, v)), 360).to_numpy()
        truth = wide("pm25_true").to_numpy() if "pm25_true" in df else None
        return cls(times, ids, wide(pm_col).to_numpy(), met, st.loc[ids], truth)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_nodes(self) -> int:
        return len(self.station_ids)


FC_NOISE = {  # (база, рост на час заблаговременности)
    "temp": (0.6, 0.04),
    "rh": (3.0, 0.15),
    "wind_speed": (0.12, 0.006),  # мультипликативный, лог-шкала
    "wind_dir": (15.0, 0.6),
    "blh": (0.15, 0.006),  # мультипликативный
}


def noisy_forecast(var: str, values: np.ndarray, lead: np.ndarray | float, rng: np.random.Generator | None) -> np.ndarray:
    """Имитация ошибки метеопрогноза. rng=None — без шума (реальный прогноз уже подставлен в met)."""
    if rng is None:
        return values
    base, growth = FC_NOISE[var]
    sd = base + growth * np.asarray(lead, float)
    eps = rng.normal(0, 1, values.shape) * sd
    if var in ("wind_speed", "blh"):
        return values * np.exp(eps)
    if var == "wind_dir":
        return np.mod(values + eps, 360)
    if var == "rh":
        return np.clip(values + eps, 5, 100)
    return values + eps


def _rolling(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(1, w // 3))
    return getattr(r, fn)().to_numpy()


def _time_feats(times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    h = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()
    m = times.month.to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow": times.dayofweek.to_numpy().astype(float),
        "heating": ((m >= 10) | (m <= 4)).astype(float),
    }


def make_tabular(
    data: WideData,
    issue_idx: np.ndarray,
    horizons=HORIZONS,
    seed: int = 0,
    with_target: bool = True,
    noise: bool = True,
) -> pd.DataFrame:
    """Табличная выборка: строка = (момент выпуска, станция, горизонт)."""
    rng = np.random.default_rng(seed) if noise else None
    pm = pd.DataFrame(data.pm).ffill(limit=3).to_numpy()
    mean24 = _rolling(pm, 24, "mean")
    max24 = _rolling(pm, 24, "max")
    nbr = neighbour_mean(pm, data.adj)
    city = np.nanmean(pm, axis=1)
    st = data.stations
    T, N = data.n_times, data.n_nodes
    frames = []
    for h in horizons:
        I = issue_idx[(issue_idx >= 24) & (issue_idx + h < T)]
        if len(I) == 0:
            continue
        V = I + h
        f: dict[str, np.ndarray] = {}
        for k in LAGS:
            f[f"pm_lag{k}"] = pm[I - k]
        f["pm_mean24"], f["pm_max24"] = mean24[I], max24[I]
        f["pm_trend6"] = pm[I] - pm[I - 6]
        f["pm_nbr"] = nbr[I]
        f["pm_city"] = np.repeat(city[I][:, None], N, axis=1)
        for v in ("temp", "rh", "wind_speed", "blh"):
            f[f"{v}_now"] = data.met[v][I]
        archive = data.forecast_archive.window(data.times[I], data.station_ids, h) if data.forecast_archive is not None else None
        weather_rng = None if archive is not None else rng
        # Прогноз погоды на момент t+h …
        for v in ("temp", "rh", "wind_speed", "blh"):
            f[f"{v}_fc"] = noisy_forecast(v, archive[v][:,-1] if archive is not None else data.met[v][V], h, weather_rng)
        wd = noisy_forecast("wind_dir", archive['wind_dir'][:,-1] if archive is not None else data.met["wind_dir"][V], h, weather_rng)
        f["wd_sin_fc"], f["wd_cos_fc"] = np.sin(np.radians(wd)), np.cos(np.radians(wd))
        # … и агрегаты прогноза по окну (t, t+h]: пришёл ли ветер, поднялась ли «крышка».
        ws_mean = _rolling(data.met["wind_speed"], h, "mean")[V]
        ws_max = _rolling(data.met["wind_speed"], h, "max")[V]
        blh_min = _rolling(data.met["blh"], h, "min")[V]
        blh_mean = _rolling(data.met["blh"], h, "mean")[V]
        t_min = _rolling(data.met["temp"], h, "min")[V]
        if archive is not None:
            ws_mean, ws_max = archive['wind_speed'].mean(axis=1), archive['wind_speed'].max(axis=1)
            blh_min, blh_mean = archive['blh'].min(axis=1), archive['blh'].mean(axis=1)
            t_min = archive['temp'].min(axis=1)
        f["ws_mean_win"] = noisy_forecast("wind_speed", ws_mean, h / 2, weather_rng)
        f["ws_max_win"] = noisy_forecast("wind_speed", ws_max, h / 2, weather_rng)
        f["blh_min_win"] = noisy_forecast("blh", blh_min, h / 2, weather_rng)
        f["blh_mean_win"] = noisy_forecast("blh", blh_mean, h / 2, weather_rng)
        f["temp_min_win"] = noisy_forecast("temp", t_min, h / 2, weather_rng)
        for k, v in _time_feats(data.times[V]).items():
            f[f"{k}_tgt"] = np.repeat(v[:, None], N, axis=1)
        f["horizon"] = np.full((len(I), N), float(h))
        for col in ("elevation", "private_sector", "lat", "lon", "tpp_distance_km"):
            f[col] = np.repeat(st[col].to_numpy()[None, :], len(I), axis=0)
        f["is_lowcost"] = np.repeat((st["kind"] == "lowcost").to_numpy()[None, :].astype(float), len(I), axis=0)
        df = pd.DataFrame({k: v.reshape(-1) for k, v in f.items()})
        df["issue_time"] = np.repeat(data.times[I].to_numpy(), N)
        df["valid_time"] = np.repeat(data.times[V].to_numpy(), N)
        df["station_id"] = np.tile(np.array(data.station_ids), len(I))
        if with_target:
            df["y"] = data.pm[V].reshape(-1)
            if data.truth is not None:
                df["y_true"] = data.truth[V].reshape(-1)
        frames.append(df)
    if not frames:
        raise ValueError('No valid issue times: need history and full forecast horizon')
    return pd.concat(frames, ignore_index=True)


META_COLS = ["issue_time", "valid_time", "station_id", "y", "y_true"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


# ---------- последовательности для ST-GNN ----------

HIST_FEATS = ["pm", "temp", "rh", "wind_speed", "wd_sin", "wd_cos", "blh", "hour_sin", "hour_cos"]
FUT_FEATS = ["temp", "rh", "wind_speed", "wd_sin", "wd_cos", "blh", "hour_sin", "hour_cos", "lead"]


def _scale_met(var: str, x: np.ndarray) -> np.ndarray:
    if var == "temp":
        return x / 15.0
    if var == "rh":
        return (x - 60) / 20.0
    if var == "wind_speed":
        return np.log1p(x) - 1.0
    if var == "blh":
        return (np.log(np.clip(x, 20, None)) - 6.0)
    return x


def make_sequences(data: WideData, issue_idx: np.ndarray, seed: int = 0, hist: int = HIST_HOURS,
                   max_h: int = MAX_H, noise: bool = True):
    """Массивы для ST-GNN: X_hist (B, hist, N, F), X_fut (B, max_h, N, F'), Y (B, max_h, N)."""
    rng = np.random.default_rng(seed) if noise else None
    I = issue_idx[(issue_idx >= hist - 1) & (issue_idx + max_h < data.n_times)]
    N = data.n_nodes
    pm = pd.DataFrame(data.pm).ffill(limit=3).to_numpy()
    tf = _time_feats(data.times)
    wd = np.radians(data.met["wind_dir"])
    base = {
        "pm": np.log1p(np.nan_to_num(np.clip(pm, 0, None), nan=0.0)) / 3.0 - 1.0,
        "temp": _scale_met("temp", data.met["temp"]),
        "rh": _scale_met("rh", data.met["rh"]),
        "wind_speed": _scale_met("wind_speed", data.met["wind_speed"]),
        "wd_sin": np.sin(wd),
        "wd_cos": np.cos(wd),
        "blh": _scale_met("blh", data.met["blh"]),
        "hour_sin": np.repeat(tf["hour_sin"][:, None], N, axis=1),
        "hour_cos": np.repeat(tf["hour_cos"][:, None], N, axis=1),
    }
    hist_ix = I[:, None] - np.arange(hist - 1, -1, -1)[None, :]  # (B, hist)
    fut_ix = I[:, None] + np.arange(1, max_h + 1)[None, :]  # (B, max_h)
    Xh = np.stack([base[k][hist_ix] for k in HIST_FEATS], axis=-1).astype(np.float32)
    # Метеопрогноз с шумом, растущим с заблаговременностью.
    lead = np.arange(1, max_h + 1, dtype=float)[None, :, None]
    fut = {}
    archive = data.forecast_archive.window(data.times[I], data.station_ids, max_h) if data.forecast_archive is not None and len(I) else None
    weather_rng = None if archive is not None else rng
    for v in ("temp", "rh", "wind_speed", "blh"):
        fut[v] = _scale_met(v, noisy_forecast(v, archive[v] if archive is not None else data.met[v][fut_ix], lead, weather_rng))
    wdf = np.radians(noisy_forecast("wind_dir", archive['wind_dir'] if archive is not None else data.met["wind_dir"][fut_ix], lead, weather_rng))
    fut["wd_sin"], fut["wd_cos"] = np.sin(wdf), np.cos(wdf)
    fut["hour_sin"], fut["hour_cos"] = base["hour_sin"][fut_ix], base["hour_cos"][fut_ix]
    fut["lead"] = np.broadcast_to(lead / max_h, fut["temp"].shape)
    Xf = np.stack([fut[k] for k in FUT_FEATS], axis=-1).astype(np.float32)
    Y = data.pm[fut_ix].astype(np.float32)  # NaN = нет наблюдения, маскируется в функции потерь
    valid = np.isfinite(Xh).all(axis=(1,2,3)) & np.isfinite(Xf).all(axis=(1,2,3))
    return Xh[valid], Xf[valid], Y[valid], I[valid]
