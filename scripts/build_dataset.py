"""Сборка реального набора данных в формате проекта из выгрузок OpenAQ и Open-Meteo.

Нужен файл станций (пример — data/stations_template.csv):
id,name,lat,lon,elevation,kind,private_sector,tpp_distance_km,sensor_id
(sensor_id — из data/raw/openaq_locations.csv; kind = official | lowcost).

    python scripts/build_dataset.py --stations data/stations.csv --openaq data/raw/openaq.csv \
        --met data/raw/met_era5.csv --out data/almaty_real.csv.gz
    python scripts/train.py --data data/almaty_real.csv.gz --stations data/stations.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True)
    ap.add_argument("--openaq", required=True)
    ap.add_argument("--met", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = pd.read_csv(a.stations)
    pm = pd.read_csv(a.openaq, parse_dates=["time"]).merge(st[["id", "sensor_id", "kind"]], on="sensor_id")
    pm = pm.rename(columns={"id": "station_id"})
    pm["time"] = pm["time"].dt.floor("h")
    pm = pm.groupby(["time", "station_id", "kind"], as_index=False)["pm25"].mean()
    pm["pm25_raw"] = pm["pm25"]
    pm["pm25_ref"] = pm["pm25"].where(pm["kind"] == "official")
    met = pd.read_csv(a.met, parse_dates=["time"])
    df = met.merge(pm[["time", "station_id", "pm25_raw", "pm25_ref"]], on=["time", "station_id"], how="left")
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} строк -> {a.out}; доля часов с PM2.5: {df['pm25_raw'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
