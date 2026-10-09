from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "meta.json").exists(), reason="модели не обучены")


def test_api_forecast():
    from fastapi.testclient import TestClient

    from almatyair.api import app

    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    assert len(c.get("/api/stations").json()) >= 5
    r = c.get("/api/forecast", params={"issue_time": "2025-01-15T06:00", "model": "lightgbm"})
    assert r.status_code == 200
    js = r.json()
    assert len(js["forecast"]) == 48 * 9
    m = c.get("/api/map", params={"issue_time": "2025-01-15T06:00", "lead_h": 24, "model": "lightgbm"}).json()
    assert len(m["pm25"]) == len(m["lat"])
