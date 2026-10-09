"""Генерирует синтетический набор данных для Алматы (3 года, 9 станций, почасово).

    python scripts/make_synthetic.py --out data/sample/almaty_demo.csv.gz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from almatyair.synthetic import colocation_pairs, simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample/almaty_demo.csv.gz")
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2025-06-30 23:00")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = simulate(a.start, a.end, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round({c: 2 for c in df.columns if c != "time"}).to_csv(out, index=False)
    colocation_pairs().round(2).to_csv(out.parent / "colocation_demo.csv", index=False)
    print(f"Сохранено {len(df):,} строк → {out}")
    winter = df[df["time"].dt.month.isin([12, 1, 2])]["pm25_true"].mean()
    summer = df[df["time"].dt.month.isin([6, 7, 8])]["pm25_true"].mean()
    print(f"Средний PM2.5 зимой: {winter:.0f} мкг/м³, летом: {summer:.0f} мкг/м³")


if __name__ == "__main__":
    main()
