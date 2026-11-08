from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from project2_app.config import ExperimentSettings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    pipeline: Pipeline
    tuning_grid: dict[str, list[Any]]


def _classifier(model_key: str, params: dict[str, Any]) -> ClassifierMixin:
    factories: dict[str, type[ClassifierMixin]] = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
    }
    try:
        factory = factories[model_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_key}") from exc
    return factory(**params)


def build_model_spec(model_key: str, settings: ExperimentSettings) -> ModelSpec:
    """Build a model pipeline and prefixed tuning grid from configuration."""
    if model_key not in settings.models:
        raise ValueError(f"Model '{model_key}' is not configured.")

    cfg = settings.models[model_key]
    estimator = _classifier(model_key, dict(cfg["standard_params"]))
    steps: list[tuple[str, Any]] = []
    if bool(cfg.get("scale_features", False)):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))

    tuning_grid = {f"model__{key}": list(values) for key, values in cfg["tuning_grid"].items()}
    return ModelSpec(
        key=model_key,
        display_name=str(cfg["display_name"]),
        pipeline=Pipeline(steps=steps),
        tuning_grid=tuning_grid,
    )


def available_models(settings: ExperimentSettings) -> dict[str, str]:
    return {key: str(value["display_name"]) for key, value in settings.models.items()}
