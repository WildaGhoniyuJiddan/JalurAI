from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="JalurAI API", version="0.1.0")


class ShipmentInput(BaseModel):
    origin_city: str
    dest_city: str
    weight: float = Field(gt=0)
    volume: float = Field(gt=0)
    carrier_type: str
    value_of_goods: float = Field(ge=0)


class PredictionResponse(BaseModel):
    risk_score: float
    risk_category: Literal["Normal", "Risiko Tinggi"]
    estimated_extra_cost: float
    estimated_delay_days: int
    shap_features: list[dict[str, str | float]]
    resolution_narrative: str
    recommended_action: str
    created_at: datetime


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: ShipmentInput) -> PredictionResponse:
    # Placeholder pipeline boundary: XGBoost, SHAP, and LLM resolver plug in here.
    distance_signal = 0.35 if shipment.origin_city.lower() != shipment.dest_city.lower() else 0.05
    weight_signal = min(shipment.weight / 1000, 0.35)
    risk_score = round(min(distance_signal + weight_signal, 0.99), 2)
    high_risk = risk_score >= 0.5
    return PredictionResponse(
        risk_score=risk_score,
        risk_category="Risiko Tinggi" if high_risk else "Normal",
        estimated_extra_cost=round(risk_score * 250000, 2),
        estimated_delay_days=2 if high_risk else 0,
        shap_features=[
            {"feature": "Rute berbeda kota", "impact": distance_signal},
            {"feature": "Berat kiriman", "impact": weight_signal},
        ],
        resolution_narrative=("Pesanan perlu ditinjau karena sinyal rute dan berat meningkatkan risiko."
                              if high_risk else "Pesanan berada dalam rentang risiko normal."),
        recommended_action=("Evaluasi armada atau ekspedisi alternatif sebelum diproses."
                            if high_risk else "Lanjutkan proses pengiriman sesuai prosedur normal."),
        created_at=datetime.now(timezone.utc),
    )
