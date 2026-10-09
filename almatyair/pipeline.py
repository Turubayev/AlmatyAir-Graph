"""Сквозной конвейер: данные → калибровка → обучение → сравнение с базовыми моделями → прогноз."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .calibration import calibrate_network
from .features import (FUT_FEATS, HIST_FEATS, HORIZONS, MAX_H, WideData, feature_columns, make_sequences,
                       make_tabular)
from .models.baselines import Persistence, QuantileGBM
from .stations import stations_frame


@dataclass
class Split:
    train_end: str = "2023-07-01"
    val_end: str = "2024-07-01"


def load_long(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df


def prepare(df: pd.DataFrame, stations: pd.DataFrame | None = None,
            colocation: pd.DataFrame | str | Path | None = None, calibration_end=None) -> tuple[WideData, dict]:
    st = stations if stations is not None else stations_frame()
    if isinstance(colocation, (str, Path)):
        colocation = pd.read_csv(colocation)
    if "pm25" not in df:
        df, calib = calibrate_network(df, st, colocation, calibration_end=calibration_end)
    else:
        calib = {}
    return WideData.from_long(df, st), calib


def split_indices(data: WideData, split: Split, step: int = 3, seed: int = 0):
    """Моменты выпуска прогноза для обучения/валидации/теста.

    Берётся каждый `step`-й час в среднем, но в случайные часы суток: при регулярной сетке
    (0, 3, 6 … ч) модель не видела бы выпусков в остальные часы и хуже работала бы на них.
    """
    t = data.times
    if step < 1 or pd.Timestamp(split.train_end) >= pd.Timestamp(split.val_end):
        raise ValueError('Invalid chronological split')
    if step == 1:
        idx = np.arange(data.n_times)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(data.n_times, data.n_times // step, replace=False))
    target_end = t[idx] + pd.Timedelta(hours=MAX_H)
    tr = idx[target_end < pd.Timestamp(split.train_end)]
    va = idx[(t[idx] >= pd.Timestamp(split.train_end)) & (target_end < pd.Timestamp(split.val_end))]
    te = idx[t[idx] >= pd.Timestamp(split.val_end)]
    return tr, va, te


def _stgnn_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def train(data: WideData, out_dir: str | Path, split: Split | None = None, epochs: int = 15,
          n_estimators: int = 600, use_stgnn: bool = True, verbose: bool = True) -> dict:
    split = split or Split()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = split_indices(data, split)

    if verbose:
        print(f"Выборки (моменты выпуска): train={len(tr)}, val={len(va)}, test={len(te)}", flush=True)
    tab_tr = make_tabular(data, tr, seed=1)
    feats = feature_columns(tab_tr)
    gbm = QuantileGBM(feats, n_estimators=n_estimators).fit(tab_tr)
    margin = gbm.conformalize(make_tabular(data, va, seed=4))
    gbm.save(out / "lightgbm.pkl")
    if verbose:
        print("LightGBM обучен; топ признаков:", ", ".join(gbm.feature_importance().head(8).index), flush=True)
        print("Конформная поправка интервала (log1p) по горизонтам:", {int(k): round(v, 3) for k, v in margin.items()}, flush=True)

    trained = {"lightgbm": True, "stgnn": False}
    if use_stgnn and _stgnn_available():
        from .models.stgnn import STGNNForecaster

        Xh, Xf, Y, _ = make_sequences(data, tr, seed=2)
        Vh, Vf, VY, _ = make_sequences(data, va, seed=3)
        net = STGNNForecaster(data.n_nodes, data.adj, len(HIST_FEATS), len(FUT_FEATS))
        net.fit(Xh, Xf, Y, epochs=epochs, val=(Vh, Vf, VY), verbose=verbose)
        net.save(out / "stgnn.pt")
        trained["stgnn"] = True

    meta = {
        "station_ids": data.station_ids,
        "horizons": list(HORIZONS),
        "split": split.__dict__,
        "trained": trained,
        "train_period": [str(data.times[0]), split.train_end],
        "calibration_period": [split.train_end, split.val_end],
        "weather_mode": "issued_forecast_archive" if data.forecast_archive is not None else "reanalysis_plus_noise_research_only",
        "split_purge_hours": MAX_H,
        "training_revision": 2,
        "operational_validation": False,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def evaluate(data: WideData, model_dir: str | Path, split: Split | None = None, cams: pd.DataFrame | None = None) -> dict:
    """Сравнение моделей на тестовом периоде. cams — длинная таблица time, station_id, pm25_cams."""
    model_dir = Path(model_dir)
    split = split or Split()
    _, _, te = split_indices(data, split, step=1)  # тест — каждый час, чтобы точно ловить эпизоды
    tab = make_tabular(data, te, seed=11)
    preds: dict[str, dict] = {}
    preds["persistence"] = {"q50": Persistence().predict(tab)}
    gbm = QuantileGBM.load(model_dir / "lightgbm.pkl")
    q = gbm.predict_quantiles(tab)
    preds["lightgbm"] = {"q10": q[0.1], "q50": q[0.5], "q90": q[0.9]}

    if (model_dir / "stgnn.pt").exists() and _stgnn_available():
        from .models.stgnn import STGNNForecaster

        net = STGNNForecaster.load(model_dir / "stgnn.pt")
        Xh, Xf, _, I = make_sequences(data, te, seed=11)
        Q = net.predict_quantiles(Xh, Xf)  # (B, 48, N, 3)
        pos = {int(i): k for k, i in enumerate(I)}
        row_issue = data.times.get_indexer(pd.to_datetime(tab["issue_time"]))
        b = np.array([pos.get(int(i), -1) for i in row_issue])
        node = np.array([data.station_ids.index(s) for s in tab["station_id"]])
        h = tab["horizon"].to_numpy().astype(int) - 1
        ok = b >= 0
        st_pred = {k: np.full(len(tab), np.nan) for k in ("q10", "q50", "q90")}
        for j, k in enumerate(("q10", "q50", "q90")):
            st_pred[k][ok] = Q[b[ok], h[ok], node[ok], j]
        preds["stgnn"] = st_pred

    if cams is not None:
        c = cams.copy()
        c["time"] = pd.to_datetime(c["time"])
        m = tab[["valid_time", "station_id"]].merge(c.rename(columns={"time": "valid_time"}), how="left")
        preds["cams"] = {"q50": m["pm25_cams"].to_numpy()}

    y = tab["y"].to_numpy()
    results: dict = {"by_horizon": {}, "episodes_24h": {}, "n_test_rows": int(np.isfinite(y).sum()),
                     "weather_mode": "issued_forecast_archive" if data.forecast_archive is not None else "reanalysis_plus_noise_research_only",
                     "issued_weather_used": data.forecast_archive is not None, "operational_validation": False}
    for name, p in preds.items():
        for h in HORIZONS:
            m = (tab["horizon"] == h).to_numpy()
            lo, hi = p.get("q10"), p.get("q90")
            r = M.summary(y[m], p["q50"][m], None if lo is None else lo[m], None if hi is None else hi[m])
            results["by_horizon"].setdefault(name, {})[h] = r
        # Эпизоды смога по среднегородскому прогнозу на 24 ч.
        m = (tab["horizon"] == 24).to_numpy()
        d = pd.DataFrame({"t": tab["valid_time"][m], "y": y[m], "p": p["q50"][m]}).groupby("t").mean()
        results["episodes_24h"][name] = M.episode_timing(d["y"], d["p"])
    return results


def recalibrate(data: WideData, model_dir: str | Path, split: Split | None = None) -> dict:
    """Пересчёт конформной поправки LightGBM без переобучения (тот же валидационный период)."""
    split = split or Split()
    _, va, _ = split_indices(data, split)
    gbm = QuantileGBM.load(Path(model_dir) / "lightgbm.pkl")
    margin = gbm.conformalize(make_tabular(data, va, seed=4))
    gbm.save(Path(model_dir) / "lightgbm.pkl")
    return margin


def results_markdown(res: dict) -> str:
    lines = ["| Модель | Горизонт, ч | RMSE | MAE | Точность AQI | POD | FAR | Покрытие 80% |", "|---|---|---|---|---|---|---|---|"]
    for name, by_h in res["by_horizon"].items():
        for h, r in by_h.items():
            cov = f"{r['coverage_80']:.2f}" if "coverage_80" in r else "—"
            lines.append(f"| {name} | {h} | {r['rmse']:.1f} | {r['mae']:.1f} | {r['aqi_acc']:.2f} | {r['pod']:.2f} | {r['far']:.2f} | {cov} |")
    lines += ["", "**Эпизоды смога (> 55.5 мкг/м³, среднее по городу, прогноз на 24 ч)**", "",
              "| Модель | Эпизодов (факт / прогноз / совпало) | Ошибка начала, ч | Ошибка окончания, ч |", "|---|---|---|---|"]
    for name, e in res["episodes_24h"].items():
        lines.append(f"| {name} | {e['episodes_obs']} / {e['episodes_pred']} / {e['episodes_matched']} | "
                     f"{e['onset_mae_h']:.1f} | {e['end_mae_h']:.1f} |")
    return "\n".join(lines)


# ---------------- оперативный прогноз ----------------

class ForecastService:
    """Загружает обученные модели и строит прогноз для заданного момента выпуска."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.gbm = QuantileGBM.load(self.dir / "lightgbm.pkl")
        self.net = None
        if (self.dir / "stgnn.pt").exists() and _stgnn_available():
            from .models.stgnn import STGNNForecaster

            self.net = STGNNForecaster.load(self.dir / "stgnn.pt")

    def forecast(self, data: WideData, issue_time: pd.Timestamp, model: str = "auto", noise: bool = False) -> pd.DataFrame:
        i = int(data.times.get_indexer([pd.Timestamp(issue_time)])[0])
        if i < 0:
            raise KeyError(f"Нет данных на момент {issue_time}")
        use_net = (model in ("auto", "stgnn")) and self.net is not None
        rows = []
        if use_net:
            Xh, Xf, _, I = make_sequences(data, np.array([i]), noise=noise)
            if len(I) == 0:
                raise KeyError("Недостаточно истории (нужно 48 ч) или прогноза погоды (48 ч)")
            Q = self.net.predict_quantiles(Xh, Xf)[0]  # (48, N, 3)
            for h in range(MAX_H):
                for n, sid in enumerate(data.station_ids):
                    rows.append((sid, h + 1, *Q[h, n]))
            name = "stgnn"
        else:
            tab = make_tabular(data, np.array([i]), horizons=range(1, MAX_H + 1), with_target=False, noise=noise)
            q = self.gbm.predict_quantiles(tab)
            for k in range(len(tab)):
                rows.append((tab["station_id"].iat[k], int(tab["horizon"].iat[k]), q[0.1][k], q[0.5][k], q[0.9][k]))
            name = "lightgbm"
        df = pd.DataFrame(rows, columns=["station_id", "lead_h", "q10", "q50", "q90"])
        df["valid_time"] = pd.Timestamp(issue_time) + pd.to_timedelta(df["lead_h"], unit="h")
        df["aqi"] = [M.aqi_label(v) for v in df["q50"]]
        df["model"] = name
        if data.truth is not None or np.isfinite(data.pm).any():
            V = data.times.get_indexer(df["valid_time"])
            nodes = [data.station_ids.index(s) for s in df["station_id"]]
            df["observed"] = [data.pm[v, n] if v >= 0 else np.nan for v, n in zip(V, nodes)]
        return df


def alerts(fc: pd.DataFrame, threshold: float = M.EPISODE_THRESHOLD, min_hours: int = 6) -> list[dict]:
    """Ожидаемые эпизоды смога по каждой станции (медиана выше порога ≥ min_hours подряд)."""
    out = []
    for sid, g in fc.groupby("station_id"):
        s = g.set_index("valid_time")["q50"]
        for start, end in M.find_episodes(s, threshold, min_hours):
            seg = g[(g["valid_time"] >= start) & (g["valid_time"] <= end)]
            out.append({
                "station_id": sid,
                "start": str(start),
                "end": str(end),
                "peak_q50": round(float(seg["q50"].max()), 1),
                "peak_q90": round(float(seg["q90"].max()), 1),
                "prob_note": "интервал q10–q90 — 80%-й диапазон неопределённости",
            })
    return out
