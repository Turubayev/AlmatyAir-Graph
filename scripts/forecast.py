"""Прогноз из командной строки.

    python scripts/forecast.py --issue-time 2025-01-15T06:00        # по демо-архиву
    python scripts/forecast.py --live                               # Open-Meteo + OpenAQ/CAMS
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from almatyair.pipeline import ForecastService, alerts, load_long, prepare  # noqa: E402
from almatyair.stations import stations_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--data", default="data/sample/almaty_demo.csv.gz")
    ap.add_argument("--issue-time")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="auto", choices=["auto", "stgnn", "lightgbm"])
    a = ap.parse_args()
    if not a.live and not a.issue_time:
        ap.error("укажите --issue-time или --live")
    svc = ForecastService(a.models)
    if a.live:
        from almatyair.live import build_live

        data, t0, src = build_live(stations_frame())
    else:
        coloc = Path(a.data).parent / "colocation_demo.csv"
        data, _ = prepare(load_long(a.data), colocation=coloc if coloc.exists() else None)
        t0, src = pd.Timestamp(a.issue_time), "demo"
    fc = svc.forecast(data, t0, model=a.model)
    print(f"Выпуск {t0} ({src}, модель {fc['model'].iat[0]}), медиана PM2.5, мкг/м³:")
    table = fc[fc["lead_h"].isin([12, 24, 36, 48])].pivot(index="station_id", columns="lead_h", values="q50").round(0)
    print(table.to_string())
    for al in alerts(fc):
        print(f"! {al['station_id']}: эпизод смога {al['start']} – {al['end']}, пик ≈ {al['peak_q50']} (до {al['peak_q90']}) мкг/м³")


if __name__ == "__main__":
    main()
