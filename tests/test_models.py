import pytest

from project2_app.config import load_config
from project2_app.models import available_models, build_model_spec


@pytest.mark.parametrize(
    "model_key",
    ["logistic_regression", "decision_tree", "random_forest", "svm", "knn"],
)
def test_all_models_build(model_key: str) -> None:
    settings = load_config()
    spec = build_model_spec(model_key, settings)
    assert spec.key == model_key
    assert "model" in spec.pipeline.named_steps
    assert spec.tuning_grid


def test_available_models_are_human_readable() -> None:
    settings = load_config()
    models = available_models(settings)
    assert models["random_forest"] == "Random Forest"
    assert models["svm"] == "Support Vector Machine"
