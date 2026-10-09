"""Базовые модели: «завтра как сегодня» и градиентный бустинг (LightGBM) с квантилями."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


class Persistence:
    """Прогноз = последнее известное значение в тот же час суток («завтра как сегодня»).
    Горизонты 24/48 ч → значение в момент выпуска, 12/36 ч → значение 12 ч назад."""

    name = "persistence"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        h = df["horizon"].to_numpy()
        same_hour = np.where(h % 24 == 0, df["pm_lag0"], df["pm_lag12"])
        return np.where(np.isfinite(same_hour), same_hour, df["pm_mean24"]).astype(float)


def _make_regressor(alpha: float, n_estimators: int, seed: int):
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
    except ImportError:  # запасной вариант без LightGBM
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=n_estimators, learning_rate=0.05, random_state=seed)


class QuantileGBM:
    """Три модели (q10, q50, q90) на общем наборе признаков. Цель — log1p(PM2.5)."""

    name = "lightgbm"

    def __init__(self, features: list[str], n_estimators: int = 600, seed: int = 0):
        self.features = features
        self.n_estimators = n_estimators
        self.seed = seed
        self.models: dict[float, object] = {}

    def fit(self, df: pd.DataFrame, target: str = "y") -> "QuantileGBM":
        d = df[np.isfinite(df[target])]
        X, y = d[self.features], np.log1p(d[target].clip(lower=0))
        for q in QUANTILES:
            self.models[q] = _make_regressor(q, self.n_estimators, self.seed).fit(X, y)
        return self

    def conformalize(self, df: pd.DataFrame, target: str = "y", coverage: float = 0.8) -> dict[float, float]:
        """Конформная калибровка интервала (CQR) на отложенной выборке, отдельно для каждого горизонта.

        Квантильный бустинг обычно даёт слишком узкие интервалы, поэтому интервал расширяется
        на поправку (в шкале log1p), при которой доля попаданий на валидации равна `coverage`.
        """
        d = df[np.isfinite(df[target])]
        X, y = d[self.features], np.log1p(d[target].clip(lower=0)).to_numpy()
        lo, hi = self.models[QUANTILES[0]].predict(X), self.models[QUANTILES[-1]].predict(X)
        score = np.maximum(lo - y, y - hi)
        h = d["horizon"].to_numpy()
        self.margin = {}
        for hv in np.unique(h):
            s = score[h == hv]
            n = len(s)
            self.margin[float(hv)] = float(np.quantile(s, min(1.0, np.ceil((n + 1) * coverage) / n)))
        return self.margin

    def _margin_for(self, horizons: np.ndarray) -> np.ndarray:
        m = getattr(self, "margin", 0.0)
        if not isinstance(m, dict):  # модели, сохранённые до версии с поправкой по горизонтам
            return np.full(len(horizons), float(m))
        keys = np.array(sorted(m))
        vals = np.array([m[k] for k in keys])
        return np.interp(horizons, keys, vals)  # для промежуточных горизонтов — интерполяция

    def predict_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        X = df[self.features]
        m = self._margin_for(df["horizon"].to_numpy(float))
        shift = {QUANTILES[0]: -m, QUANTILES[-1]: m}
        out = {q: np.expm1(mdl.predict(X) + shift.get(q, 0.0)) for q, mdl in self.models.items()}
        # Квантили не должны пересекаться.
        stack = np.sort(np.stack([out[q] for q in QUANTILES]), axis=0)
        return {q: np.clip(stack[i], 0, None) for i, q in enumerate(QUANTILES)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(df)[0.5]

    def feature_importance(self) -> pd.Series:
        m = self.models[0.5]
        imp = getattr(m, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "QuantileGBM":
        with open(path, "rb") as f:
            return pickle.load(f)
