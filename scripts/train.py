"""Обучение LightGBM и ST-GNN + сравнение с базовыми моделями на тестовом периоде.

    python scripts/train.py --data data/sample/almaty_demo.csv.gz --out models
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from almatyair.pipeline import Split, evaluate, load_long, prepare, recalibrate, results_markdown, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample/almaty_demo.csv.gz")
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--train-end", default="2023-07-01")
    ap.add_argument("--val-end", default="2024-07-01")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--no-stgnn", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="не обучать: пересчитать калибровку интервала и сравнение")
    ap.add_argument("--stations", help="CSV станций (id,name,lat,lon,elevation,kind,private_sector,tpp_distance_km)")
    ap.add_argument("--colocation", default="data/sample/colocation_demo.csv",
                    help="совместные измерения датчик/станция для калибровки ('' — калибровать по соседям)")
    ap.add_argument("--cams", help="CSV с колонками time, station_id, pm25_cams (scripts/fetch_openmeteo.py --cams)")
    a = ap.parse_args()

    split = Split(a.train_end, a.val_end)
    st = pd.read_csv(a.stations).set_index("id") if a.stations else None
    coloc = a.colocation if a.colocation and Path(a.colocation).exists() else None
    data, calib = prepare(load_long(a.data), st, coloc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    for sid, v in calib.items():
        print(f"Калибровка {sid}: b={v['b']:.2f}, κ={v['kappa']:.2f}, эталон {v['reference']} ({v['distance_km']} км)")

    if a.eval_only:
        print("Поправка интервала:", recalibrate(data, out, split))
        meta = {"trained": {"stgnn": False}}
    else:
        meta = train(data, out, split, epochs=a.epochs, n_estimators=a.trees, use_stgnn=not a.no_stgnn)
    if meta["trained"]["stgnn"]:
        try:  # ONNX для ONNX Runtime (сервер без PyTorch, мобильные клиенты)
            from almatyair.models.stgnn import STGNNForecaster

            STGNNForecaster.load(out / "stgnn.pt").export_onnx(out / "stgnn.onnx")
            print("Экспорт:", out / "stgnn.onnx")
        except Exception as e:  # noqa: BLE001
            print("ONNX-экспорт пропущен:", e)
    cams = pd.read_csv(a.cams) if a.cams else None
    res = evaluate(data, out, split, cams=cams)
    rd = Path(a.results)
    rd.mkdir(exist_ok=True)
    (rd / "benchmark.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    md = results_markdown(res)
    (rd / "benchmark.md").write_text(f"# Сравнение моделей (тест: с {a.val_end})\n\n{md}\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
