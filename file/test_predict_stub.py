"""Smoke test for the order-dataset-backed prediction API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

response = client.get("/health")
assert response.status_code == 200, f"Health failed: {response.text}"
assert response.json()["status"] == "ok"
print("[PASS] /health")

response = client.post(
    "/predict",
    json={
        "origin_city": "Tangerang",
        "dest_city": "Bandung",
        "dest_island": "JAWA",
        "weight": 2.5,
        "value_of_goods": 500000,
        "distance_km": 150,
        "estimated_shipping_cost": 15000,
        "qty": 2,
        "jumlah_kategori": 1,
        "tier_layanan": "Reguler",
        "kurir": "JNE",
    },
)
assert response.status_code == 200, response.text
data = response.json()
assert "risk_score" in data
assert "risk_category" in data
assert len(data["shap_features"]) == 3
assert "resolution_narrative" in data
assert "recommended_action" in data
assert data["risk_category"] in ("Normal", "Risiko Tinggi")
print(
    "[PASS] /predict (Java route): "
    f"risk={data['risk_score']}, cat={data['risk_category']}"
)

response = client.post(
    "/predict",
    json={
        "origin_city": "Tangerang",
        "dest_city": "Medan",
        "dest_island": "SUMATERA",
        "weight": 25.0,
        "value_of_goods": 2500000,
        "distance_km": 1800,
        "estimated_shipping_cost": 900000,
        "qty": 1,
        "jumlah_kategori": 1,
        "tier_layanan": "Hemat",
        "kurir": "SPX",
    },
)
assert response.status_code == 200, response.text
data = response.json()
assert len(data["shap_features"]) == 3
assert data["risk_category"] in ("Normal", "Risiko Tinggi")
print(
    "[PASS] /predict (cross-island route): "
    f"risk={data['risk_score']}, cat={data['risk_category']}"
)

print("\nAll smoke tests passed.")
