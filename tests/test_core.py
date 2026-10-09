import numpy as np
import pandas as pd
import pytest

from almatyair import metrics as M
from almatyair.calibration import HumidityCalibration
from almatyair.features import WideData, feature_columns, make_sequences, make_tabular
from almatyair.graph import static_adjacency, wind_adjacency
from almatyair.models.baselines import Persistence, QuantileGBM
from almatyair.stations import stations_frame
from almatyair.synthetic import colocation_pairs, simulate


@pytest.fixture(scope="module")
def small():
    df = simulate("2023-11-01", "2024-01-31 23:00", seed=1)
    df["pm25"] = df["pm25_raw"]
    return WideData.from_long(df, stations_frame())


def test_synthetic_has_winter_smog():
    df = simulate("2024-01-01", "2024-01-31 23:00", seed=3)
    assert df["pm25_true"].mean() > 20
    assert set(df["station_id"]) == set(stations_frame().index)


def test_calibration_recovers_humidity_growth():
    co = colocation_pairs(n_days=60)
    cal = HumidityCalibration().fit(co["pm25_raw"], co["rh"], co["pm25_ref"])
    corr = cal.correct(co["pm25_raw"], co["rh"])
    err_raw = np.median(np.abs(co["pm25_raw"] - co["pm25_ref"]) / co["pm25_ref"])
    err_cal = np.median(np.abs(corr - co["pm25_ref"]) / co["pm25_ref"])
    assert err_cal < err_raw / 3
    assert cal.kappa > 0


def test_adjacency_prefers_same_elevation():
    st = stations_frame()
    a = static_adjacency(st)
    assert np.allclose(np.diag(a), 0)
    ids = list(st.index)
    # Каменское плато (1250 м) слабо связано с низинным Турксибом.
    assert a[ids.index("KAM"), ids.index("TRK")] < a[ids.index("ALM"), ids.index("ZHT")]


def test_wind_adjacency_is_directional():
    st = stations_frame()
    ids = list(st.index)
    south_wind = wind_adjacency(st, wind_dir_deg=180)  # дует с юга на север
    i, j = ids.index("TRK"), ids.index("BOS")  # TRK севернее BOS
    assert south_wind[i, j] > south_wind[j, i]


def test_metrics():
    y = np.array([10, 60, 70, 20, 80])
    p = np.array([12, 50, 75, 60, 90])
    pod, far = M.pod_far(y, p)
    assert pod == pytest.approx(2 / 3)
    assert far == pytest.approx(1 / 3)
    assert M.aqi_label(5) == "Хорошо"
    assert M.aqi_label(300) == "Опасно"


def test_episode_detection():
    t = pd.date_range("2024-01-01", periods=48, freq="h")
    s = pd.Series(10.0, index=t)
    s.iloc[10:20] = 100
    s.iloc[14] = 40  # короткий провал внутри эпизода склеивается
    ep = M.find_episodes(s)
    assert len(ep) == 1 and ep[0][0] == t[10]


def test_tabular_and_gbm(small):
    idx = np.arange(48, small.n_times - 49, 6)
    tab = make_tabular(small, idx)
    feats = feature_columns(tab)
    assert "wind_speed_fc" in feats and "y" not in feats
    gbm = QuantileGBM(feats, n_estimators=30).fit(tab)
    q = gbm.predict_quantiles(tab.head(200))
    assert np.all(q[0.1] <= q[0.5]) and np.all(q[0.5] <= q[0.9])
    assert np.isfinite(Persistence().predict(tab)).mean() > 0.9


def test_sequences_shapes(small):
    Xh, Xf, Y, I = make_sequences(small, np.arange(0, small.n_times, 24))
    assert Xh.shape[1:] == (48, small.n_nodes, 9)
    assert Xf.shape[1:] == (48, small.n_nodes, 9)
    assert Y.shape == (len(I), 48, small.n_nodes)
    assert np.isfinite(Xh).all() and np.isfinite(Xf).all()


def test_stgnn_forward(small):
    pytest.importorskip("torch")
    from almatyair.models.stgnn import STGNNForecaster

    Xh, Xf, Y, _ = make_sequences(small, np.arange(48, 600, 12))
    net = STGNNForecaster(small.n_nodes, small.adj, Xh.shape[-1], Xf.shape[-1], hidden=16)
    net.fit(Xh, Xf, Y, epochs=1, verbose=False)
    Q = net.predict_quantiles(Xh[:2], Xf[:2])
    assert Q.shape == (2, 48, small.n_nodes, 3)
    assert (Q[..., 0] <= Q[..., 1] + 1e-4).all() and (Q[..., 1] <= Q[..., 2] + 1e-4).all()
