from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    confusion: np.ndarray
    report: pd.DataFrame
    predictions: pd.DataFrame
    roc_points: pd.DataFrame


def _positive_scores(model: BaseEstimator, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_test), dtype=float)
        minimum, maximum = scores.min(), scores.max()
        if maximum == minimum:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - minimum) / (maximum - minimum)
    return np.asarray(model.predict(X_test), dtype=float)


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    y_pred = np.asarray(model.predict(X_test), dtype=int)
    y_true = y_test.to_numpy(dtype=int)
    scores = _positive_scores(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malignant"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    predictions = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "malignant_score": scores,
        },
        index=X_test.index,
    ).sort_index()
    return EvaluationResult(metrics, confusion, report, predictions, roc_points)


def confusion_explanation(confusion: np.ndarray) -> str:
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    return (
        f"The model correctly identified {tp} malignant cases and {tn} benign cases. "
        f"It incorrectly classified {fn} malignant case(s) as benign and {fp} benign case(s) "
        "as malignant. Because malignant is the positive class, recall is especially useful for "
        "understanding how well the model avoids false negatives."
    )


def metric_table(metrics: dict[str, float]) -> pd.DataFrame:
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall (Malignant)",
        "f1": "F1 Score",
        "roc_auc": "ROC-AUC",
    }
    return pd.DataFrame(
        [{"Metric": labels[key], "Value": value} for key, value in metrics.items()]
    )
