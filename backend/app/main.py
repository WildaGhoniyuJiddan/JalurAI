from datetime import datetime, timezone
from typing import Literal

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predict import (
    FEATURE_COLUMNS,
    MODEL_LOAD_ERROR,
    build_feature_vector,
    classifier,
    explainer,
    regressor,
    resolve_with_llm,
    scale_features,
)


app = FastAPI(title="JalurAI API", version="0.1.0")


class ShipmentInput(BaseModel):
    origin_city: str
    dest_city: str
    weight: float = Field(gt=0)
    volume: float = Field(gt=0)
    value_of_goods: float = Field(ge=0)
    distance_km: float = Field(ge=0)
    carrier_type: str
    armada_type: str = "truk"


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
        or regressor is None
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
        risk_category = (
            "Risiko Tinggi" if risk_score >= 0.5 else "Normal"
        )

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
            extra_cost_pct = max(
                float(regressor.predict([scaled])[0]), 0.0
            )
            estimated_extra_cost = round(
                raw_features["base_shipping_cost_rp"]
                * extra_cost_pct
                / 100,
                2,
            )
            delay_days = round(extra_cost_pct / 25, 1)
        else:
            estimated_extra_cost = 0.0
            delay_days = 0.0

        narrative, action = resolve_with_llm(
            risk_score, shap_top3, delay_days
        )

        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_category=risk_category,
            estimated_extra_cost=estimated_extra_cost,
            estimated_delay_days=delay_days,
            shap_features=[
                ShapFeature(**feature) for feature in shap_top3
            ],
            resolution_narrative=narrative,
            recommended_action=action,
            created_at=datetime.now(timezone.utc),
        )
    except (ValueError, TypeError, xgb.core.XGBoostError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediksi model JalurAI gagal: {exc}",
        ) from exc
