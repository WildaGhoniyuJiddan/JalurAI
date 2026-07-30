"""Smoke test for the JalurAI model-backed prediction API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)

response = client.get("/health")
assert response.status_code == 200, f"Health failed: {response.text}"
assert response.json()["status"] == "ok"
print("[PASS] /health")

response = client.post(
    "/predict",
    json={
        "origin_city": "Jakarta",
        "dest_city": "Bandung",
        "weight": 5.0,
        "volume": 0.05,
        "value_of_goods": 500000,
        "distance_km": 150,
        "carrier_type": "JNE",
        "armada_type": "truk",
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
    "[PASS] /predict (Normal route): "
    f"risk={data['risk_score']}, cat={data['risk_category']}"
)

response = client.post(
    "/predict",
    json={
        "origin_city": "Jakarta",
        "dest_city": "Medan",
        "weight": 25.0,
        "volume": 0.8,
        "value_of_goods": 2500000,
        "distance_km": 1800,
        "carrier_type": "J&T Express",
        "armada_type": "truk",
    },
)
assert response.status_code == 200, response.text
data = response.json()
assert len(data["shap_features"]) == 3
assert data["risk_category"] in ("Normal", "Risiko Tinggi")
print(
    "[PASS] /predict (Long cross-island): "
    f"risk={data['risk_score']}, cat={data['risk_category']}"
)

print("\nAll smoke tests passed.")
