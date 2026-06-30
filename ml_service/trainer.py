"""
XGBoost trainer for food freshness prediction.
Uses shared FeatureBuilder to eliminate train/serve skew.
"""

import json
import logging
import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .feature_builder import CONFIG, FeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "donations", "food_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "freshness_xgb_v1.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "feature_builder_v1.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_v1.json")


def load_data() -> pd.DataFrame:
    """Load and validate training data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} samples from {DATA_PATH}")

    # Validate required columns
    required = CONFIG.NUMERICAL_FEATURES + CONFIG.CATEGORICAL_FEATURES + [CONFIG.TARGET_COL]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Drop rows with missing values
    initial = len(df)
    df = df.dropna(subset=required)
    if len(df) < initial:
        logger.info(f"Dropped {initial - len(df)} rows with missing values")

    return df


def train_model(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Train XGBoost model with FeatureBuilder.
    Returns training metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = load_data()
    logger.info(f"Training on {len(df)} samples")

    # Feature engineering
    builder = FeatureBuilder()
    X = builder.fit_transform(df)
    y = builder.encode_target(df[CONFIG.TARGET_COL])

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target classes: {builder.classes}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train XGBoost
    n_classes = len(builder.classes)
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )

    logger.info("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=builder.classes,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    metrics = {
        "accuracy": report["accuracy"],
        "roc_auc_weighted": auc,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1-score": report[cls]["f1-score"],
                "support": int(report[cls]["support"]),
            }
            for cls in builder.classes
        },
        "confusion_matrix": cm.tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X.columns),
    }

    # Save model
    model.save_model(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save FeatureBuilder
    joblib.dump(builder, ENCODER_PATH)
    logger.info(f"FeatureBuilder saved to {ENCODER_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {METRICS_PATH}")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"ROC-AUC (weighted): {metrics['roc_auc_weighted']:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=builder.classes))

    return metrics


if __name__ == "__main__":
    metrics = train_model()
    print(f"\nFinal Accuracy: {metrics['accuracy']:.2%}")
    print(f"Final ROC-AUC: {metrics['roc_auc_weighted']:.4f}")
