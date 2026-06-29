"""
Model evaluation script.
Generates precision/recall, confusion matrix, and ROC-AUC metrics.
Run standalone: python -m ml_service.evaluate
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from .feature_builder import FeatureBuilder, CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'donations', 'food_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'freshness_xgb_v1.json')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'feature_builder_v1.pkl')
EVAL_OUTPUT = os.path.join(BASE_DIR, 'models', 'evaluation_report.json')


def evaluate() -> dict:
    """Run full evaluation on test set."""
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # Load or rebuild FeatureBuilder
    if os.path.exists(ENCODER_PATH):
        builder = joblib.load(ENCODER_PATH)
    else:
        builder = FeatureBuilder()
        builder.fit(df)

    X = builder.transform(df)
    y = builder.encode_target(df[CONFIG.TARGET_COL])
    classes = builder.classes

    # Train/test split (same as training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC
    try:
        auc_weighted = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc_weighted = auc_macro = 0.0

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )

    evaluation = {
        'overall': {
            'accuracy': report['accuracy'],
            'roc_auc_weighted': round(auc_weighted, 4),
            'roc_auc_macro': round(auc_macro, 4),
            'test_samples': len(y_test),
        },
        'per_class': {},
        'confusion_matrix': {
            'labels': list(classes),
            'matrix': cm.tolist(),
        },
        'classification_report': report,
    }

    for i, cls in enumerate(classes):
        evaluation['per_class'][cls] = {
            'precision': round(float(precision[i]), 4),
            'recall': round(float(recall[i]), 4),
            'f1_score': round(float(f1[i]), 4),
            'support': int(support[i]),
        }

    # Save report
    os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
    with open(EVAL_OUTPUT, 'w') as f:
        json.dump(evaluation, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"ECOFEAST FRESHNESS MODEL — EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Accuracy:       {report['accuracy']:.4f}")
    print(f"ROC-AUC (wtd):  {auc_weighted:.4f}")
    print(f"ROC-AUC (macro): {auc_macro:.4f}")
    print(f"Test samples:   {len(y_test)}")
    print(f"\nPer-Class Metrics:")
    print(f"{'-'*60}")
    for cls in classes:
        m = evaluation['per_class'][cls]
        print(f"  {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1_score']:.3f}  N={m['support']}")
    print(f"\nConfusion Matrix:")
    print(f"{'-'*60}")
    print(f"  Predicted >>  {'  '.join(classes)}")
    for i, cls in enumerate(classes):
        print(f"  {cls:12s}  {'  '.join(f'{v:4d}' for v in cm[i])}")
    print(f"\nFull report saved to: {EVAL_OUTPUT}")

    return evaluation


if __name__ == '__main__':
    evaluate()
