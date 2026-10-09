"""Загрузка реальной метеорологии (ERA5 или архив прогнозов) и PM2.5 CAMS по координатам станций.

    python scripts/fetch_openmeteo.py --start 2022-10-01 --end 2025-04-30 --out data/raw/met_era5.csv
    python scripts/fetch_openmeteo.py --start 2024-10-01 --end 2025-04-30 --cams --out data/raw/cams.csv

Результат объединяется с измерениями OpenAQ в scripts/build_dataset.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from almatyair.data import openmeteo  # noqa: E402
from almatyair.stations import stations_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stations", help="CSV станций (по умолчанию — встроенный список)")
    ap.add_argument("--cams", action="store_true", help="вместо метео скачать PM2.5 CAMS (базовая модель)")
    ap.add_argument("--source", default="era5", choices=["era5", "forecast"])
    a = ap.parse_args()
    st = pd.read_csv(a.stations).set_index("id") if a.stations else stations_frame()
    frames = []
    for sid, s in st.iterrows():
        print("->", sid, s["name"], flush=True)
        if a.cams:
            df = openmeteo.cams_pm25(s["lat"], s["lon"], a.start, a.end)
        else:
            df = openmeteo.weather_history(s["lat"], s["lon"], a.start, a.end, source=a.source)
        df["station_id"] = sid
        frames.append(df)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out, index=False)
    print("Сохранено:", out)


if __name__ == "__main__":
    main()
