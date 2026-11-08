from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from project2_app.config import ExperimentSettings
from project2_app.data import SplitBundle
from project2_app.models import ModelSpec


@dataclass(frozen=True)
class TrainingResult:
    estimator: BaseEstimator
    best_params: dict[str, Any]
    cv_mean: float
    cv_std: float
    mode: str


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("model__"): value for key, value in params.items()}


def train_model(
    spec: ModelSpec,
    split: SplitBundle,
    settings: ExperimentSettings,
    mode: str,
) -> TrainingResult:
    """Train a standard or modestly tuned model with stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )

    if mode == "Tuned":
        search = GridSearchCV(
            estimator=spec.pipeline,
            param_grid=spec.tuning_grid,
            scoring=settings.scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(split.X_train, split.y_train)
        return TrainingResult(
            estimator=search.best_estimator_,
            best_params=_clean_params(search.best_params_),
            cv_mean=float(search.best_score_),
            cv_std=float(search.cv_results_["std_test_score"][search.best_index_]),
            mode=mode,
        )

    scores = cross_val_score(
        spec.pipeline,
        split.X_train,
        split.y_train,
        scoring=settings.scoring,
        cv=cv,
        n_jobs=-1,
    )
    spec.pipeline.fit(split.X_train, split.y_train)
    model_params = spec.pipeline.named_steps["model"].get_params(deep=False)
    configured_keys = set(spec.tuning_grid)
    standard_params = {
        key.removeprefix("model__"): model_params[key.removeprefix("model__")]
        for key in configured_keys
        if key.removeprefix("model__") in model_params
    }
    return TrainingResult(
        estimator=spec.pipeline,
        best_params=standard_params,
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores)),
        mode="Standard",
    )
