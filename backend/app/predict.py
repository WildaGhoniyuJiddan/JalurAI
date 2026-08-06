import json
import os
from pathlib import Path
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


MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
CLASSIFIER_PATH = MODEL_DIR / "xgb_classifier.json"
DELAY_REGRESSOR_PATH = MODEL_DIR / "xgb_delay_regressor.json"
COST_REGRESSOR_PATH = MODEL_DIR / "xgb_cost_regressor.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

classifier: xgb.XGBClassifier | None = None
delay_regressor: xgb.XGBRegressor | None = None
cost_regressor: xgb.XGBRegressor | None = None
explainer: Any = None
metadata: dict[str, Any] = {}
MODEL_LOAD_ERROR: str | None = None

required_artifacts = [
    CLASSIFIER_PATH,
    DELAY_REGRESSOR_PATH,
    COST_REGRESSOR_PATH,
    METADATA_PATH,
]
missing_artifacts = [
    str(path) for path in required_artifacts if not path.is_file()
]

if missing_artifacts:
    MODEL_LOAD_ERROR = (
        "Model artifact tidak ditemukan: " + ", ".join(missing_artifacts)
    )
else:
    try:
        classifier = xgb.XGBClassifier()
        classifier.load_model(str(CLASSIFIER_PATH))

        delay_regressor = xgb.XGBRegressor()
        delay_regressor.load_model(str(DELAY_REGRESSOR_PATH))

        cost_regressor = xgb.XGBRegressor()
        cost_regressor.load_model(str(COST_REGRESSOR_PATH))

        with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        explainer = shap.TreeExplainer(classifier)
    except (OSError, ValueError, KeyError, xgb.core.XGBoostError) as exc:
        MODEL_LOAD_ERROR = f"Gagal memuat model JalurAI: {exc}"


FEATURE_COLUMNS = metadata.get("feature_columns", [])
CATEGORICAL_MAPPINGS = metadata.get("categorical_mappings", {})
SCALER_MEAN = metadata.get("scaler_mean", [])
SCALER_STD = metadata.get("scaler_std", [])

OUTER_JAWA_CITIES = {
    "medan",
    "palembang",
    "makassar",
    "banjarmasin",
    "pontianak",
    "manado",
    "padang",
    "pekanbaru",
    "jambi",
    "bengkulu",
    "lampung",
    "balikpapan",
    "samarinda",
    "ternate",
    "ambon",
    "mataram",
    "kupang",
    "papua",
}

RESOLVER_SYSTEM_PROMPT = """
Kamu adalah Resolver Agent untuk sistem prediksi risiko pengiriman logistik
JalurAI. Tugasmu menerima data prediksi dan menghasilkan narasi penjelasan
serta rekomendasi tindakan dalam Bahasa Indonesia.

Input yang kamu terima dalam JSON:
- risk_score: skor risiko gabungan model XGBoost antara 0 dan 1
- risk_category: "Risiko Tinggi" atau "Normal"
- shap_features: tiga faktor risiko teratas dari SHAP
- estimated_extra_cost: estimasi kelebihan biaya dalam Rupiah
- estimated_delay_days: estimasi keterlambatan dalam hari
- origin_city dan dest_city: kota asal dan tujuan
- distance_km: jarak pengiriman dalam kilometer
- tier_layanan: tier layanan pengiriman
- kurir: kurir yang dipilih

Tugasmu:
1. Jelaskan hasil prediksi dengan bahasa sederhana untuk staf gudang.
2. Berikan rekomendasi spesifik yang dapat langsung dilakukan.
3. Untuk risiko tinggi, rekomendasi harus berupa tindakan konkret.
4. Untuk risiko normal, tetap berikan saran optimasi yang relevan.
5. Jangan membuat angka, rute, kurir, atau kondisi baru yang tidak ada pada input.

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


def encode_category(field: str, value: str) -> int:
    mapping = CATEGORICAL_MAPPINGS.get(field, {})
    if value in mapping:
        return int(mapping[value])
    fallback = mapping.get("LAINNYA", 0)
    return int(fallback)


def is_outer_jawa(shipment: Any) -> int:
    island = str(getattr(shipment, "dest_island", "") or "").upper()
    if island:
        return int(island != "JAWA")
    destination = str(getattr(shipment, "dest_city", "")).lower()
    return int(destination in OUTER_JAWA_CITIES)


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
    """Build the same pre-dispatch features used by ``file/train.py``."""
    weight = max(float(shipment.weight), 0.01)
    shipping_cost = shipment.estimated_shipping_cost
    if shipping_cost is None:
        shipping_cost = shipment.distance_km * 2000

    return {
        "weight_kg": weight,
        "qty": max(float(shipment.qty), 1.0),
        "jumlah_kategori": max(float(shipment.jumlah_kategori), 1.0),
        "berat_2_4kg": float(2 <= weight < 4),
        "luar_jawa": float(is_outer_jawa(shipment)),
        "nilai_barang_idr": max(float(shipment.value_of_goods), 0.0),
        "jarak_tempuh_km": max(float(shipment.distance_km), 0.0),
        "ongkir_per_kg": max(float(shipping_cost), 0.0) / weight,
        "rasio_jarak_per_kg": max(float(shipment.distance_km), 0.0) / weight,
        "tier_layanan": float(
            encode_category("tier_layanan", shipment.tier_layanan)
        ),
        "kurir": float(encode_category("kurir", shipment.kurir)),
    }


def resolve_with_llm(
    risk_score: float,
    shap_features: list[dict[str, str | float]],
    delay_days: float,
    extra_cost_amount: float = 0.0,
    origin_city: str = "",
    dest_city: str = "",
    distance_km: float = 0.0,
    tier_layanan: str = "",
    kurir: str = "",
) -> tuple[str, str]:
    """Resolve a prediction through local Ollama with a safe fallback."""
    risk_category = "Risiko Tinggi" if risk_score >= 0.5 else "Normal"
    payload = {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "shap_features": shap_features,
        "estimated_extra_cost": extra_cost_amount,
        "estimated_delay_days": delay_days,
        "origin_city": origin_city,
        "dest_city": dest_city,
        "distance_km": distance_km,
        "tier_layanan": tier_layanan,
        "kurir": kurir,
    }

    if OLLAMA_AVAILABLE and ollama is not None:
        try:
            response = ollama.generate(
                model="llama3.2",
                prompt=(
                    "Analisis data prediksi logistik berikut dan kembalikan "
                    "JSON sesuai format yang diminta:\n\n"
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

    return _resolve_with_llm_stub(risk_score, shap_features, delay_days)


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
            f"Pesanan ini dikategorikan berisiko tinggi (skor {risk_score:.2f}). "
            f"Faktor utama adalah {primary['feature']} "
            f"(dampak: {float(primary['impact']):+.4f}). "
            f"Estimasi keterlambatan {delay_days:.1f} hari."
        )
        action = "Evaluasi ulang tier layanan atau kurir sebelum diproses."
    else:
        narrative = (
            f"Pesanan berada dalam rentang risiko normal (skor {risk_score:.2f}). "
            f"Faktor dominan: {primary['feature']} "
            f"(dampak: {float(primary['impact']):+.4f})."
        )
        action = "Lanjutkan proses pengiriman sesuai prosedur normal."

    return narrative, action
