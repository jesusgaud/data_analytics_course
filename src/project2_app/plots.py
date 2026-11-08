from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def save_confusion_matrix(confusion: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(confusion)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix",
    )
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(roc_points: pd.DataFrame, auc: float, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(roc_points["fpr"], roc_points["tpr"], label=f"Model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _model_step(estimator: Any) -> Any:
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["model"]
    return estimator


def save_model_specific_plot(
    estimator: Any,
    feature_names: tuple[str, ...],
    model_key: str,
    destination: Path,
) -> str | None:
    model = _model_step(estimator)
    values: np.ndarray | None = None
    title: str | None = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        title = "Feature Importance"
    elif model_key == "logistic_regression" and hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
        title = "Absolute Logistic Regression Coefficients"

    if values is None or title is None:
        return None

    top_n = min(12, len(feature_names))
    order = np.argsort(values)[-top_n:]
    labels = np.asarray(feature_names)[order]
    selected = values[order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, selected)
    ax.set_title(title)
    ax.set_xlabel("Relative importance")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return title
