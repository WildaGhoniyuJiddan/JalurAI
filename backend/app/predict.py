import json
import os
from typing import Any

if os.name == "nt":
    os.environ.setdefault(
        "WINDIR", os.environ.get("SystemRoot", "C:\\Windows")
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(
            os.environ.get("TEMP", os.path.dirname(__file__)),
            "jalurai-matplotlib",
        ),
    )

import shap
import xgboost as xgb

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False


MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "xgb_classifier.json")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "xgb_regressor.json")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

classifier: xgb.XGBClassifier | None = None
regressor: xgb.XGBRegressor | None = None
explainer: Any = None
metadata: dict[str, Any] = {}
MODEL_LOAD_ERROR: str | None = None

required_artifacts = [CLASSIFIER_PATH, REGRESSOR_PATH, METADATA_PATH]
missing_artifacts = [
    path for path in required_artifacts if not os.path.isfile(path)
]

if missing_artifacts:
    MODEL_LOAD_ERROR = (
        "Model artifact tidak ditemukan: " + ", ".join(missing_artifacts)
    )
else:
    try:
        classifier = xgb.XGBClassifier()
        classifier.load_model(CLASSIFIER_PATH)

        regressor = xgb.XGBRegressor()
        regressor.load_model(REGRESSOR_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        explainer = shap.TreeExplainer(classifier)
    except (OSError, ValueError, KeyError, xgb.core.XGBoostError) as exc:
        MODEL_LOAD_ERROR = f"Gagal memuat model JalurAI: {exc}"


FEATURE_COLUMNS = metadata.get("feature_columns", [])
LABEL_ENCODER_CLASSES = metadata.get("label_encoder_classes", {})
SCALER_MEAN = metadata.get("scaler_mean", [])
SCALER_STD = metadata.get("scaler_std", [])

RESOLVER_SYSTEM_PROMPT = """
Kamu adalah Resolver Agent untuk sistem prediksi risiko pengiriman logistik
JalurAI. Tugasmu menerima data prediksi dan menghasilkan narasi penjelasan
serta rekomendasi tindakan dalam Bahasa Indonesia.

Input yang kamu terima dalam JSON:
- risk_score: skor risiko model XGBoost antara 0 dan 1
- risk_category: "Normal" atau "Risiko Tinggi"
- shap_features: tiga faktor risiko teratas dari SHAP
- estimated_extra_cost: estimasi kelebihan biaya dalam Rupiah
- estimated_delay_days: estimasi keterlambatan dalam hari
- origin_city dan dest_city: kota asal dan tujuan
- distance_km: jarak pengiriman dalam kilometer
- carrier_type: jenis ekspedisi
- armada_type: truk, kapal_laut, pesawat, atau motor

Tugasmu:
1. Jelaskan hasil prediksi dengan bahasa sederhana untuk staf gudang.
2. Berikan rekomendasi spesifik yang dapat langsung dilakukan.
3. Untuk risiko tinggi, rekomendasi harus berupa tindakan konkret.
4. Untuk risiko normal, tetap berikan saran optimasi yang relevan.
5. Jangan membuat angka, rute, ekspedisi, atau kondisi baru yang tidak ada
   pada input.

Kembalikan JSON valid saja tanpa teks tambahan:
{
  "resolution_narrative": "<narasi Bahasa Indonesia>",
  "recommended_action": "<rekomendasi tindakan spesifik>",
  "confidence": "<tinggi|sedang|rendah>",
  "reasoning": "<alasan singkat>"
}
"""

RESOLVER_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "resolution_narrative": {"type": "string"},
        "recommended_action": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["tinggi", "sedang", "rendah"],
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "resolution_narrative",
        "recommended_action",
        "confidence",
        "reasoning",
    ],
}


def encode_armada(armada: str) -> int:
    mapping = LABEL_ENCODER_CLASSES
    if isinstance(mapping.get("armada_type"), dict):
        mapping = mapping["armada_type"]
    return int(mapping.get(armada, 0))


def scale_features(raw: dict[str, float]) -> list[float]:
    """Apply the StandardScaler parameters stored during training."""
    if not (
        len(FEATURE_COLUMNS) == len(SCALER_MEAN) == len(SCALER_STD)
    ):
        raise ValueError("Metadata scaler tidak sesuai dengan feature_columns")

    return [
        (raw.get(feature, 0.0) - SCALER_MEAN[index])
        / SCALER_STD[index]
        if SCALER_STD[index] > 0
        else raw.get(feature, 0.0)
        for index, feature in enumerate(FEATURE_COLUMNS)
    ]


def build_feature_vector(shipment: Any) -> dict[str, float]:
    """Build raw features in the same order and shape used during training."""
    java_cities = {
        "jakarta",
        "bandung",
        "surabaya",
        "semarang",
        "yogyakarta",
        "solo",
        "malang",
        "blitar",
        "tangerang",
        "bekasi",
        "depok",
        "bogor",
    }
    jawa_origin = int(shipment.origin_city.lower() in java_cities)
    jawa_dest = int(shipment.dest_city.lower() in java_cities)
    cross_island = int(jawa_origin != jawa_dest)
    base_shipping_cost = round(shipment.distance_km * 2000, 0)
    cost_value_ratio = base_shipping_cost / max(
        shipment.value_of_goods, 10000
    )
    dist_per_kg = shipment.distance_km / max(shipment.weight, 0.1)

    return {
        "weight_kg": shipment.weight,
        "volume_m3": shipment.volume,
        "value_goods_rp": shipment.value_of_goods,
        "distance_km": shipment.distance_km,
        "base_shipping_cost_rp": base_shipping_cost,
        "cost_value_ratio": round(cost_value_ratio, 4),
        "dist_per_kg": round(dist_per_kg, 2),
        "jawa_origin": jawa_origin,
        "jawa_dest": jawa_dest,
        "cross_island": cross_island,
        "armada_type": encode_armada(shipment.armada_type),
    }


def resolve_with_llm(
    risk_score: float,
    shap_features: list[dict[str, str | float]],
    delay_days: float,
    extra_cost_amount: float = 0.0,
    origin_city: str = "",
    dest_city: str = "",
    distance_km: float = 0.0,
    carrier_type: str = "",
    armada_type: str = "",
) -> tuple[str, str]:
    """Resolve a prediction through local Ollama with a safe fallback."""
    risk_category = (
        "Risiko Tinggi" if risk_score >= 0.5 else "Normal"
    )
    payload = {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "shap_features": shap_features,
        "estimated_extra_cost": extra_cost_amount,
        "estimated_delay_days": delay_days,
        "origin_city": origin_city,
        "dest_city": dest_city,
        "distance_km": distance_km,
        "carrier_type": carrier_type,
        "armada_type": armada_type,
    }

    if OLLAMA_AVAILABLE and ollama is not None:
        try:
            response = ollama.generate(
                model="llama3.2",
                prompt=(
                    "Analisis data prediksi logistik berikut dan "
                    "kembalikan JSON sesuai format yang diminta:\n\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
                system=RESOLVER_SYSTEM_PROMPT.strip(),
                options={"temperature": 0.3, "num_predict": 500},
                format=RESOLVER_RESPONSE_FORMAT,
            )
            result_text = (
                response.get("response", "")
                if isinstance(response, dict)
                else getattr(response, "response", "")
            )
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
                narrative = result.get("resolution_narrative")
                action = result.get("recommended_action")
                if (
                    isinstance(narrative, str)
                    and narrative.strip()
                    and isinstance(action, str)
                    and action.strip()
                ):
                    return narrative.strip(), action.strip()
        except Exception:
            pass

    return _resolve_with_llm_stub(
        risk_score, shap_features, delay_days
    )


def _resolve_with_llm_stub(
    risk_score: float,
    shap_features: list[dict[str, str | float]],
    delay_days: float,
) -> tuple[str, str]:
    """Deterministic fallback when Ollama cannot resolve a prediction."""
    top_features = sorted(
        shap_features,
        key=lambda item: abs(float(item["impact"])),
        reverse=True,
    )
    primary = (
        top_features[0]
        if top_features
        else {"feature": "unknown", "impact": 0.0, "direction": "netral"}
    )

    if risk_score >= 0.5:
        narrative = (
            f"Pesanan ini dikategorikan berisiko tinggi "
            f"(skor {risk_score:.2f}). Faktor utama yang berkontribusi "
            f"adalah {primary['feature']} "
            f"(dampak: {float(primary['impact']):+.4f}). "
            f"Estimasi keterlambatan {delay_days:.1f} hari dengan "
            "potensi kelebihan biaya signifikan."
        )
        action = (
            "Evaluasi armada atau ekspedisi alternatif sebelum diproses."
        )
    else:
        narrative = (
            f"Pesanan berada dalam rentang risiko normal "
            f"(skor {risk_score:.2f}). Faktor dominan: "
            f"{primary['feature']} "
            f"(dampak: {float(primary['impact']):+.4f}). "
            "Tidak ada indikasi keterlambatan atau kelebihan biaya "
            "yang signifikan."
        )
        action = "Lanjutkan proses pengiriman sesuai prosedur normal."

    return narrative, action
