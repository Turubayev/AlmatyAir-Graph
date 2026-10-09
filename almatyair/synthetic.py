"""Синтетический, но физически осмысленный генератор данных для Алматы.

Нужен, чтобы весь конвейер (калибровка → граф → модели → метрики → API) запускался
без ключей и интернета. Модель упрощённая «ящичная»: концентрация в каждом районе
растёт от выбросов, делённых на высоту слоя перемешивания, и вымывается ветром.
Учтены:
* сезонный и суточный ход температуры, отопительный сезон;
* синоптические эпизоды застоя (низкий ветер, мощная инверсия) и фронты, которые
  резко проветривают котловину — именно их должна «увидеть» модель в метеопрогнозе;
* горно-долинная циркуляция (ночью сток с гор, днём — на горы);
* выбросы: угольный частный сектор, транспорт (часы пик), ТЭЦ-2;
* высота станции: верхние районы чаще оказываются над слоем инверсии;
* недорогие оптические датчики, завышающие PM2.5 при высокой влажности.

Данные НЕ являются реальными измерениями и используются только для отладки и демо.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stations import STATIONS, Station, bearing_deg, haversine_km

MET_COLS = ["temp", "rh", "wind_speed", "wind_dir", "blh"]


def _ou(n: int, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Процесс Орнштейна–Уленбека (почасовой шаг) — синоптическая изменчивость."""
    a = np.exp(-1.0 / tau)
    out = np.empty(n)
    out[0] = 0.0
    eps = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    for i in range(1, n):
        out[i] = a * out[i - 1] + eps[i]
    return out


def simulate_weather(times: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(times)
    doy = times.dayofyear.to_numpy()
    hour = times.hour.to_numpy()
    season = -np.cos(2 * np.pi * (doy - 20) / 365.25)  # -1 зима … +1 лето

    synoptic = _ou(n, tau=72, sigma=1.0, rng=rng)  # >0 — антициклон/застой
    # Фронты: в среднем раз в 6 суток зимой, резкий ветер и похолодание.
    front = np.zeros(n)
    t = 0
    while t < n:
        t += int(rng.exponential(6 * 24)) + 24
        if t >= n:
            break
        dur = int(rng.integers(10, 30))
        amp = rng.uniform(3.0, 8.0)
        shape = amp * np.exp(-np.arange(dur) / (dur / 2.5))
        front[t : t + dur] = np.maximum(front[t : t + dur], shape[: n - t])

    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)
    temp = 10 + 15 * season + (4 + 3 * (season + 1) / 2) * diurnal + 3 * synoptic - 0.8 * front
    temp += rng.normal(0, 0.6, n)

    # Ветер: слабый ночью, сильнее днём; застой гасит ветер, фронт усиливает.
    ws = (1.3 + 0.7 * np.clip(diurnal, 0, None) + 0.5 * (season + 1) / 2) * np.exp(-0.35 * synoptic)
    ws = ws + front + rng.gamma(2.0, 0.15, n)
    ws = np.clip(ws, 0.2, 20)

    # Горно-долинная циркуляция: ночью с юга (с гор), днём с севера; фронт — с СЗ.
    valley_dir = np.where((hour >= 21) | (hour < 9), 180.0, 0.0)
    front_w = np.clip(front / 4, 0, 1)
    wd = (1 - front_w) * valley_dir + front_w * 315.0 + rng.normal(0, 25, n)
    wd = np.mod(wd, 360)

    rh = 55 - 18 * season - 12 * diurnal + 5 * synoptic + 10 * front_w + rng.normal(0, 5, n)
    rh = np.clip(rh, 15, 98)

    # Высота слоя перемешивания: днём конвекция (летом сильнее), ночью инверсия.
    conv = np.clip(diurnal, 0, None) * (300 + 1500 * (season + 1) / 2)
    blh = 120 + conv + 90 * ws - 70 * synoptic + rng.normal(0, 30, n)
    blh = np.clip(blh, 40, 3500)

    return pd.DataFrame(
        {"temp": temp, "rh": rh, "wind_speed": ws, "wind_dir": wd, "blh": blh}, index=times
    )


def _emissions(st: Station, met: pd.DataFrame) -> np.ndarray:
    hour = met.index.hour.to_numpy()
    dow = met.index.dayofweek.to_numpy()
    heat_demand = np.clip(16 - met["temp"].to_numpy(), 0, None)
    heat_profile = 1 + 0.6 * np.exp(-((hour - 8) ** 2) / 6) + 0.9 * np.exp(-((hour - 20) ** 2) / 8)
    coal = heat_demand * heat_profile * (0.3 + 2.2 * st.private_sector)
    traffic = (
        6 + 9 * np.exp(-((hour - 8.5) ** 2) / 2) + 10 * np.exp(-((hour - 18.5) ** 2) / 3)
    ) * np.where(dow < 5, 1.0, 0.7)
    traffic = traffic * (1.3 - 0.5 * st.private_sector)
    # ТЭЦ-2 к западу от города: шлейф доходит, когда ветер дует от неё к станции.
    tpp_lat, tpp_lon = 43.285, 76.745
    to_station = bearing_deg(tpp_lat, tpp_lon, st.lat, st.lon)
    from_dir = np.mod(met["wind_dir"].to_numpy() + 180, 360)  # куда дует ветер
    align = np.clip(np.cos(np.radians(from_dir - to_station)), 0, None)
    tpp = 25 * align * np.exp(-st.tpp_distance_km / 8) * (0.4 + 0.6 * (heat_demand > 0))
    return coal + traffic + tpp


def simulate(
    start: str = "2021-07-01",
    end: str = "2025-06-30 23:00",
    stations: list[Station] | None = None,
    seed: int = 42,
    missing_rate: float = 0.02,
) -> pd.DataFrame:
    """Почасовые данные в «длинном» формате: time, station_id, pm25, pm25_raw, метео."""
    rng = np.random.default_rng(seed)
    stations = stations or STATIONS
    times = pd.date_range(start, end, freq="h")
    met = simulate_weather(times, rng)
    n, m = len(times), len(stations)

    elev = np.array([s.elevation for s in stations])
    lat = np.array([s.lat for s in stations])
    lon = np.array([s.lon for s in stations])
    E = np.stack([_emissions(s, met) for s in stations], axis=1)  # (n, m)

    # Перенос между районами: вес от расстояния и согласованности с ветром.
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    brg = bearing_deg(lat[:, None], lon[:, None], lat[None, :], lon[None, :])  # brg[i, j]: азимут i→j

    ws = met["wind_speed"].to_numpy()
    wd = met["wind_dir"].to_numpy()
    blh = met["blh"].to_numpy()
    C = np.zeros((n, m))
    c = np.full(m, 15.0)
    base_elev = elev.min()
    for t in range(n):
        h_eff = max(blh[t], 40.0)
        above = np.exp(-np.clip(elev - base_elev, 0, None) / (0.9 * h_eff + 60))
        lam = 0.015 + 0.02 * ws[t] + 0.00025 * h_eff
        src = 120.0 * E[t] / (h_eff + 150.0) * above
        to_dir = (wd[t] + 180) % 360  # направление, куда дует ветер
        # w[i, j]: насколько воздух из j приносится в i
        align = np.clip(np.cos(np.radians(brg.T - to_dir)), 0, None)  # brg.T[i,j] = азимут j→i
        w = align * np.exp(-d / 6.0) * min(ws[t], 6) * 0.02
        np.fill_diagonal(w, 0)
        adv = (w * (c[None, :] - c[:, None])).sum(axis=1)
        c = c * np.exp(-lam) + src + adv
        c = np.clip(c, 0, None)
        C[t] = c

    background = 4 + 2 * np.sin(2 * np.pi * times.dayofyear.to_numpy() / 365.25)[:, None]
    true_pm = C + background
    obs_noise = rng.lognormal(0, 0.08, (n, m))
    pm = true_pm * obs_noise

    # Оптические датчики: завышение при высокой влажности (гигроскопический рост).
    rh = met["rh"].to_numpy()[:, None] / 100
    growth = 1 + 0.15 * rh**2 / (1 - np.minimum(rh, 0.95))
    kinds = np.array([s.kind for s in stations])
    gain = rng.uniform(1.1, 1.5, m)
    raw = np.where(kinds[None, :] == "lowcost", pm * growth * gain[None, :], pm)
    raw = raw * rng.lognormal(0, 0.05, (n, m))

    # Пропуски (сбои связи), блоками по несколько часов.
    mask = np.zeros((n, m), bool)
    for j in range(m):
        k = int(missing_rate * n / 6)
        starts = rng.integers(0, n - 12, k)
        for s0 in starts:
            mask[s0 : s0 + rng.integers(1, 12), j] = True
    raw[mask] = np.nan

    rows = []
    for j, s in enumerate(stations):
        df = met.copy()
        # Небольшие локальные отличия метео по высоте.
        df["temp"] = df["temp"] - 0.0045 * (s.elevation - base_elev) + rng.normal(0, 0.3, n)
        df["station_id"] = s.id
        df["pm25_true"] = true_pm[:, j]
        df["pm25_raw"] = raw[:, j]
        # «Официальное» значение есть только у станций Казгидромета.
        df["pm25_ref"] = np.where(s.kind == "official", raw[:, j], np.nan)
        rows.append(df)
    out = pd.concat(rows).rename_axis("time").reset_index()
    out = out[["time", "station_id", "pm25_raw", "pm25_ref", "pm25_true", *MET_COLS]]
    return out.sort_values(["time", "station_id"]).reset_index(drop=True)


def colocation_pairs(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Синтетическая кампания совместного размещения датчика рядом с эталонной станцией."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    ref = rng.lognormal(3.2, 0.9, n)
    rh = np.clip(rng.normal(65, 18, n), 15, 98)
    r = rh / 100
    raw = ref * (1 + 0.15 * r**2 / (1 - np.minimum(r, 0.95))) * 1.3 * rng.lognormal(0, 0.07, n) + 1.5
    return pd.DataFrame({"pm25_raw": raw, "rh": rh, "pm25_ref": ref})
