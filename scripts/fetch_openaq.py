"""Загрузка измерений PM2.5 из OpenAQ v3 (нужен OPENAQ_API_KEY).

    python scripts/fetch_openaq.py --list                       # станции в радиусе 25 км
    python scripts/fetch_openaq.py --start 2022-10-01 --end 2025-04-30 --out data/raw/openaq.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from almatyair.data.openaq import OpenAQ  # noqa: E402
from almatyair.stations import ALMATY_CENTER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="data/raw/openaq.csv")
    a = ap.parse_args()
    api = OpenAQ()
    locs = api.locations(*ALMATY_CENTER)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    locs.to_csv("data/raw/openaq_locations.csv", index=False)
    print(locs[["sensor_id", "name", "provider", "is_monitor", "lat", "lon", "last"]].to_string())
    if a.list:
        return
    if not (a.start and a.end):
        ap.error("укажите --start и --end")
    frames = []
    for _, r in locs.iterrows():
        print("->", r["sensor_id"], r["name"], flush=True)
        df = api.sensor_hours(int(r["sensor_id"]), a.start, a.end)
        if df.empty:
            continue
        df["sensor_id"] = r["sensor_id"]
        frames.append(df)
    pd.concat(frames).to_csv(a.out, index=False)
    print("Сохранено:", a.out)


if __name__ == "__main__":
    main()
