"""Train JalurAI models from the real-order feature dataset.

The order attributes come from ``jalurai_orders.csv``.  The delay and
over-cost labels in that file are simulated because distributor event
timestamps are unavailable; they are still kept as explicit targets rather
than being treated as observed operational outcomes.

Only features available before dispatch are used.  In particular,
``prob_telat``, ``label_telat``, ``hari_keterlambatan``,
``label_kelebihan_biaya``, and ``nilai_kelebihan_biaya_idr`` never enter the
feature matrix.
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "jalurai_orders.csv"
MODELS_DIR = ROOT_DIR / "models"

FEATURE_COLUMNS = [
    "weight_kg",
    "qty",
    "jumlah_kategori",
    "berat_2_4kg",
    "luar_jawa",
    "nilai_barang_idr",
    "jarak_tempuh_km",
    "ongkir_per_kg",
    "rasio_jarak_per_kg",
    "tier_layanan",
    "kurir",
]

TIER_MAPPING = {
    "Instan": 0,
    "Same Day": 1,
    "Next Day": 2,
    "Reguler": 3,
    "Hemat": 4,
    "Kargo": 5,
}
KURIR_MAPPING = {
    "SPX": 0,
    "JNE": 1,
    "J&T": 2,
    "GOSEND": 3,
    "GRAB": 4,
    "LAINNYA": 5,
}


def as_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode(mapping: dict[str, int], value: str) -> int:
    return mapping.get(value, mapping["LAINNYA"] if "LAINNYA" in mapping else 0)


def feature_row(row: dict[str, str]) -> list[float]:
    weight = max(as_float(row, "berat_kg"), 0.01)
    return [
        weight,
        max(as_float(row, "qty"), 1.0),
        max(as_float(row, "jumlah_kategori"), 1.0),
        float(2.0 <= weight < 4.0),
        as_float(row, "luar_jawa"),
        max(as_float(row, "nilai_barang_idr"), 0.0),
        max(as_float(row, "jarak_tempuh_km"), 0.0),
        max(as_float(row, "ongkir_per_kg"), 0.0),
        max(as_float(row, "rasio_jarak_per_kg"), 0.0),
        float(encode(TIER_MAPPING, row.get("tier_layanan", "Reguler"))),
        float(encode(KURIR_MAPPING, row.get("kurir", "LAINNYA"))),
    ]


def load_data() -> tuple[
    list[list[float]], list[int], list[int], list[float], list[float], int
]:
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    active_rows = [row for row in rows if row.get("is_cancelled") == "0"]
    if not active_rows:
        raise ValueError("jalurai_orders.csv tidak memiliki order aktif")

    features = [feature_row(row) for row in active_rows]
    delay_targets = [int(row.get("label_telat", 0)) for row in active_rows]
    cost_targets = [
        int(row.get("label_kelebihan_biaya", 0)) for row in active_rows
    ]
    delay_days = [as_float(row, "hari_keterlambatan") for row in active_rows]
    extra_cost = [
        max(as_float(row, "nilai_kelebihan_biaya_idr"), 0.0)
        for row in active_rows
    ]
    high_risk_targets = [
        int(delay or cost) for delay, cost in zip(delay_targets, cost_targets)
    ]

    return (
        features,
        high_risk_targets,
        delay_targets,
        delay_days,
        extra_cost,
        len(rows),
    )


def make_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )


def make_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def main() -> None:
    (
        features,
        high_risk_targets,
        delay_targets,
        delay_days,
        extra_cost,
        source_rows,
    ) = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)

    indices = list(range(len(features)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=high_risk_targets,
    )
    x_train = [features[index] for index in train_indices]
    x_test = [features[index] for index in test_indices]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_high_train = [high_risk_targets[index] for index in train_indices]
    y_high_test = [high_risk_targets[index] for index in test_indices]
    classifier = make_classifier()
    classifier.fit(x_train_scaled, y_high_train)
    class_predictions = classifier.predict(x_test_scaled)
    class_probabilities = classifier.predict_proba(x_test_scaled)[:, 1]
    classifier_auc = roc_auc_score(y_high_test, class_probabilities)
    classifier_report = classification_report(y_high_test, class_predictions)
    classifier.save_model(MODELS_DIR / "xgb_classifier.json")

    delay_train_indices = [
        index for index in train_indices if delay_targets[index] == 1
    ]
    delay_test_indices = [
        index for index in test_indices if delay_targets[index] == 1
    ]
    delay_regressor = make_regressor()
    delay_regressor.fit(
        scaler.transform([features[index] for index in delay_train_indices]),
        [delay_days[index] for index in delay_train_indices],
    )
    delay_predictions = delay_regressor.predict(
        scaler.transform([features[index] for index in delay_test_indices])
    )
    delay_mae = mean_absolute_error(
        [delay_days[index] for index in delay_test_indices], delay_predictions
    )
    delay_r2 = r2_score(
        [delay_days[index] for index in delay_test_indices], delay_predictions
    )
    delay_regressor.save_model(MODELS_DIR / "xgb_delay_regressor.json")

    cost_train_indices = [
        index for index in train_indices if extra_cost[index] > 0
    ]
    cost_test_indices = [
        index for index in test_indices if extra_cost[index] > 0
    ]
    cost_regressor = make_regressor()
    cost_regressor.fit(
        scaler.transform([features[index] for index in cost_train_indices]),
        [extra_cost[index] for index in cost_train_indices],
    )
    cost_predictions = cost_regressor.predict(
        scaler.transform([features[index] for index in cost_test_indices])
    )
    cost_mae = mean_absolute_error(
        [extra_cost[index] for index in cost_test_indices], cost_predictions
    )
    cost_r2 = r2_score(
        [extra_cost[index] for index in cost_test_indices], cost_predictions
    )
    cost_regressor.save_model(MODELS_DIR / "xgb_cost_regressor.json")

    metadata = {
        "dataset": "jalurai_orders.csv",
        "source_rows": source_rows,
        "training_rows": len(features),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_mappings": {
            "tier_layanan": TIER_MAPPING,
            "kurir": KURIR_MAPPING,
        },
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_std": [float(value) for value in scaler.scale_],
        "targets": {
            "classifier": "label_telat OR label_kelebihan_biaya",
            "delay_regressor": "hari_keterlambatan where label_telat=1",
            "cost_regressor": "nilai_kelebihan_biaya_idr where label_kelebihan_biaya=1",
        },
        "metrics": {
            "classifier_auc_roc": float(classifier_auc),
            "delay_mae_days": float(delay_mae),
            "delay_r2": float(delay_r2),
            "cost_mae_idr": float(cost_mae),
            "cost_r2": float(cost_r2),
        },
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with (MODELS_DIR / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("=== JalurAI Model Training (real-order dataset) ===")
    print(
        f"Samples aktif: {len(features)} "
        f"(train={len(train_indices)}, test={len(test_indices)})"
    )
    print("[Composite Risk Classifier: telat OR over-cost]")
    print(f"AUC-ROC: {classifier_auc:.4f}")
    print(classifier_report)
    print("[Delay Regressor]")
    print(f"MAE: {delay_mae:.4f} hari | R2: {delay_r2:.4f}")
    print("[Over-cost Regressor]")
    print(f"MAE: Rp {cost_mae:,.2f} | R2: {cost_r2:.4f}")
    print(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
