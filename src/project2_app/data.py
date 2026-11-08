from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from project2_app.config import ExperimentSettings


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    feature_names: tuple[str, ...]
    target_labels: tuple[str, ...]


@dataclass(frozen=True)
class SplitBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset() -> DatasetBundle:
    """Load the packaged Breast Cancer Wisconsin Diagnostic dataset.

    scikit-learn encodes malignant=0 and benign=1. We remap the target so
    malignant=1, making the medically important class the positive class.
    """
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    original_target = dataset.target.astype(int)
    y = (original_target == 0).astype(int)
    y.name = "malignant"

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=tuple(str(name) for name in dataset.feature_names),
        target_labels=("benign", "malignant"),
    )


def split_dataset(dataset: DatasetBundle, settings: ExperimentSettings) -> SplitBundle:
    """Create the fixed stratified holdout split shared by every model."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=dataset.y,
    )
    return SplitBundle(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
