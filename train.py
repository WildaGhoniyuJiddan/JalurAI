import csv
import json
import os
from datetime import datetime, timezone

from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


DATA_PATH = "synthetic_shipments.csv"
MODELS_DIR = "E:/AIC/models"

FEATURE_COLUMNS = [
    "weight_kg",
    "volume_m3",
    "value_goods_rp",
    "distance_km",
    "base_shipping_cost_rp",
    "cost_value_ratio",
    "dist_per_kg",
    "extra_cost_amount_rp",
    "extra_cost_pct",
    "jawa_origin",
    "jawa_dest",
    "cross_island",
    "armada_type",
]

ARMADA_TYPE_MAPPING = {
    "truk": 0,
    "kapal_laut": 1,
    "pesawat": 2,
    "motor": 3,
}


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    features = [
        [
            float(row[column])
            if column != "armada_type"
            else ARMADA_TYPE_MAPPING[row[column]]
            for column in FEATURE_COLUMNS
        ]
        for row in rows
    ]
    classification_targets = [int(row["is_high_risk"]) for row in rows]
    regression_targets = [float(row["extra_cost_pct"]) for row in rows]

    return features, classification_targets, regression_targets


def main():
    features, classification_targets, regression_targets = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)

    (
        x_train,
        x_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        features,
        classification_targets,
        regression_targets,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    classifier.fit(x_train_scaled, y_class_train)

    class_predictions = classifier.predict(x_test_scaled)
    class_probabilities = classifier.predict_proba(x_test_scaled)[:, 1]
    classifier_auc = roc_auc_score(y_class_test, class_probabilities)
    report = classification_report(y_class_test, class_predictions)
    classifier.save_model(os.path.join(MODELS_DIR, "xgb_classifier.json"))

    high_risk_train_indices = [
        index for index, target in enumerate(y_class_train) if target == 1
    ]
    high_risk_test_indices = [
        index for index, target in enumerate(y_class_test) if target == 1
    ]

    x_reg_train = x_train_scaled[high_risk_train_indices]
    x_reg_test = x_test_scaled[high_risk_test_indices]
    y_reg_train_high_risk = [
        y_reg_train[index] for index in high_risk_train_indices
    ]
    y_reg_test_high_risk = [
        y_reg_test[index] for index in high_risk_test_indices
    ]

    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
    )
    regressor.fit(x_reg_train, y_reg_train_high_risk)

    regression_predictions = regressor.predict(x_reg_test)
    regressor_mae = mean_absolute_error(
        y_reg_test_high_risk, regression_predictions
    )
    regressor_r2 = r2_score(y_reg_test_high_risk, regression_predictions)
    regressor.save_model(os.path.join(MODELS_DIR, "xgb_regressor.json"))

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "label_encoder_classes": ARMADA_TYPE_MAPPING,
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_std": [float(value) for value in scaler.scale_],
        "classifier_auc_roc": float(classifier_auc),
        "regressor_r2": float(regressor_r2),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("=== JalurAI Model Training ===")
    print(
        f"Samples: {len(features)} "
        f"(train={len(x_train)}, test={len(x_test)})"
    )
    print("[XGBoost Classifier]")
    print(f"AUC-ROC: {classifier_auc:.4f}")
    print("Classification Report:")
    print(report)
    print("[XGBoost Regressor]")
    print(f"MAE: {regressor_mae:.4f}")
    print(f"R2 Score: {regressor_r2:.4f}")
    print("Models saved to E:/AIC/models/")
    print("Training complete.")


if __name__ == "__main__":
    main()
