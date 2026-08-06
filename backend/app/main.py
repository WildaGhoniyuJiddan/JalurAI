from datetime import datetime, timezone
from typing import Literal

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.predict import (
    FEATURE_COLUMNS,
    MODEL_LOAD_ERROR,
    build_feature_vector,
    classifier,
    cost_regressor,
    delay_regressor,
    explainer,
    resolve_with_llm,
    scale_features,
)


app = FastAPI(title="JalurAI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ShipmentInput(BaseModel):
    origin_city: str
    dest_city: str
    dest_island: str = "JAWA"
    weight: float = Field(gt=0)
    value_of_goods: float = Field(ge=0)
    distance_km: float = Field(ge=0)
    estimated_shipping_cost: float | None = Field(default=None, ge=0)
    qty: int = Field(default=1, ge=1)
    jumlah_kategori: int = Field(default=1, ge=1)
    tier_layanan: str = "Reguler"
    kurir: str = "SPX"


class ShapFeature(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    risk_score: float
    risk_category: Literal["Normal", "Risiko Tinggi"]
    estimated_extra_cost: float
    estimated_delay_days: float
    shap_features: list[ShapFeature]
    resolution_narrative: str
    recommended_action: str
    created_at: datetime


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: ShipmentInput) -> PredictionResponse:
    if (
        MODEL_LOAD_ERROR
        or classifier is None
        or delay_regressor is None
        or cost_regressor is None
        or explainer is None
    ):
        raise HTTPException(
            status_code=503,
            detail=MODEL_LOAD_ERROR or "Model JalurAI belum siap digunakan",
        )

    try:
        raw_features = build_feature_vector(shipment)
        scaled = scale_features(raw_features)
        feature_matrix = xgb.DMatrix([scaled])

        # XGBoost's sklearn wrapper expects array-like input here; the
        # DMatrix is retained for TreeExplainer below.
        probability = classifier.predict_proba([scaled])[0]
        risk_score = float(probability[1])
        risk_category = "Risiko Tinggi" if risk_score >= 0.5 else "Normal"

        shap_values = explainer.shap_values(feature_matrix)
        if isinstance(shap_values, list):
            sample_shap_values = shap_values[-1][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            sample_shap_values = shap_values[0, :, -1]
        else:
            sample_shap_values = shap_values[0]

        shap_top3 = [
            {
                "feature": feature,
                "impact": float(impact),
                "direction": (
                    "meningkatkan_risiko"
                    if impact > 0
                    else "menurunkan_risiko"
                    if impact < 0
                    else "netral"
                ),
            }
            for feature, impact in sorted(
                zip(FEATURE_COLUMNS, sample_shap_values),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:3]
        ]

        if risk_category == "Risiko Tinggi":
            estimated_extra_cost = max(
                float(cost_regressor.predict([scaled])[0]), 0.0
            )
            delay_days = max(
                float(delay_regressor.predict([scaled])[0]), 0.0
            )
        else:
            estimated_extra_cost = 0.0
            delay_days = 0.0

        try:
            narrative, action = resolve_with_llm(
                risk_score=risk_score,
                shap_features=shap_top3,
                delay_days=delay_days,
                extra_cost_amount=estimated_extra_cost,
                origin_city=shipment.origin_city,
                dest_city=shipment.dest_city,
                distance_km=shipment.distance_km,
                tier_layanan=shipment.tier_layanan,
                kurir=shipment.kurir,
            )
        except Exception:
            narrative = "Resolver Agent tidak tersedia. Menggunakan fallback."
            action = (
                "Periksa koneksi Ollama dan pastikan model llama3.2 "
                "sudah diunduh."
            )

        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_category=risk_category,
            estimated_extra_cost=round(estimated_extra_cost, 2),
            estimated_delay_days=round(delay_days, 1),
            shap_features=[ShapFeature(**feature) for feature in shap_top3],
            resolution_narrative=narrative,
            recommended_action=action,
            created_at=datetime.now(timezone.utc),
        )
    except (ValueError, TypeError, xgb.core.XGBoostError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediksi model JalurAI gagal: {exc}",
        ) from exc
