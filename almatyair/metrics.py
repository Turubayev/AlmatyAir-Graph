"""Метрики качества прогноза PM2.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Категории AQI для PM2.5 (US EPA, редакция 2024), мкг/м³, 24-часовое среднее.
AQI_BREAKS = [9.0, 35.4, 55.4, 125.4, 225.4]
AQI_LABELS = ["Хорошо", "Умеренно", "Вредно для чувствительных групп", "Вредно", "Очень вредно", "Опасно"]
EPISODE_THRESHOLD = 55.5  # начало категории «Вредно»


def aqi_category(pm) -> np.ndarray:
    return np.digitize(np.asarray(pm, float), AQI_BREAKS, right=True)


def aqi_label(pm: float) -> str:
    return AQI_LABELS[int(aqi_category([pm])[0])]


def _clean(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def rmse(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.mean(np.abs(y - p)))


def aqi_accuracy(y, p) -> float:
    y, p = _clean(y, p)
    return float(np.mean(aqi_category(y) == aqi_category(p)))


def pod_far(y, p, threshold: float = EPISODE_THRESHOLD) -> tuple[float, float]:
    """POD — доля пойманных превышений; FAR — доля ложных тревог среди прогнозов превышения."""
    y, p = _clean(y, p)
    obs, fc = y > threshold, p > threshold
    hits = np.sum(obs & fc)
    misses = np.sum(obs & ~fc)
    false = np.sum(~obs & fc)
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = false / (hits + false) if hits + false else np.nan
    return float(pod), float(far)


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


def find_episodes(series: pd.Series, threshold: float = EPISODE_THRESHOLD, min_hours: int = 6) -> list[tuple]:
    """Эпизоды смога: непрерывные отрезки выше порога длиной ≥ min_hours (пропуски ≤ 2 ч склеиваются)."""
    s = series.sort_index()
    above = (s > threshold).to_numpy().copy()
    # Склеиваем короткие провалы (≤ 2 ч) внутри эпизода.
    true_idx = np.flatnonzero(above)
    for a, b in zip(true_idx[:-1], true_idx[1:]):
        if 1 < b - a <= 3:
            above[a:b] = True
    episodes, start = [], None
    idx = s.index
    for t, flag in zip(idx, above):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            if (t - start) >= pd.Timedelta(hours=min_hours):
                episodes.append((start, t))
            start = None
    if start is not None and (idx[-1] - start) >= pd.Timedelta(hours=min_hours):
        episodes.append((start, idx[-1]))
    return episodes


def episode_timing(obs: pd.Series, pred: pd.Series, threshold: float = EPISODE_THRESHOLD) -> dict:
    """Средняя ошибка (ч) начала и окончания эпизода для совпавших по времени эпизодов."""
    eo, ep = find_episodes(obs, threshold), find_episodes(pred, threshold)
    onset, end, matched = [], [], 0
    for s0, e0 in eo:
        cands = [(s1, e1) for s1, e1 in ep if s1 < e0 and e1 > s0]
        if not cands:
            continue
        s1, e1 = max(cands, key=lambda c: (min(c[1], e0) - max(c[0], s0)))
        onset.append((s1 - s0) / pd.Timedelta(hours=1))
        end.append((e1 - e0) / pd.Timedelta(hours=1))
        matched += 1
    return {
        "episodes_obs": len(eo),
        "episodes_pred": len(ep),
        "episodes_matched": matched,
        "onset_mae_h": float(np.mean(np.abs(onset))) if onset else np.nan,
        "end_mae_h": float(np.mean(np.abs(end))) if end else np.nan,
        "onset_bias_h": float(np.mean(onset)) if onset else np.nan,
        "end_bias_h": float(np.mean(end)) if end else np.nan,
    }


def summary(y, p, lo=None, hi=None) -> dict:
    pod, far = pod_far(y, p)
    out = {"rmse": rmse(y, p), "mae": mae(y, p), "aqi_acc": aqi_accuracy(y, p), "pod": pod, "far": far}
    if lo is not None and hi is not None:
        out["coverage_80"] = interval_coverage(y, lo, hi)
    return out
