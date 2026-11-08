from pathlib import Path

from project2_app.config import load_config
from project2_app.data import load_dataset, split_dataset
from project2_app.evaluation import evaluate_model
from project2_app.models import build_model_spec
from project2_app.training import train_model


def test_random_forest_standard_end_to_end(tmp_path: Path) -> None:
    del tmp_path
    settings = load_config()
    dataset = load_dataset()
    split = split_dataset(dataset, settings)
    spec = build_model_spec("random_forest", settings)
    trained = train_model(spec, split, settings, "Standard")
    result = evaluate_model(trained.estimator, split.X_test, split.y_test)

    assert result.metrics["accuracy"] >= 0.90
    assert result.metrics["recall"] >= 0.85
    assert result.confusion.shape == (2, 2)
    assert len(result.predictions) == len(split.X_test)
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0


def test_logistic_regression_tuned_end_to_end() -> None:
    settings = load_config()
    dataset = load_dataset()
    split = split_dataset(dataset, settings)
    spec = build_model_spec("logistic_regression", settings)
    trained = train_model(spec, split, settings, "Tuned")
    result = evaluate_model(trained.estimator, split.X_test, split.y_test)

    assert trained.mode == "Tuned"
    assert trained.best_params
    assert result.metrics["f1"] >= 0.85
