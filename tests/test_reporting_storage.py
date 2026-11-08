from __future__ import annotations

import json
from pathlib import Path

import project2_app.storage as storage
from project2_app.config import load_config
from project2_app.data import load_dataset, split_dataset
from project2_app.evaluation import confusion_explanation, evaluate_model, metric_table
from project2_app.models import build_model_spec
from project2_app.reporting import (
    best_metric_summary,
    create_run_artifacts,
    team_comparison_frame,
)
from project2_app.storage import (
    import_result_package,
    load_team_results,
    package_result,
    save_team_result,
)
from project2_app.training import train_model


def _completed_run(tmp_path: Path, monkeypatch):
    results_dir = tmp_path / "results"
    team_dir = tmp_path / "team_results"
    results_dir.mkdir()
    team_dir.mkdir()
    monkeypatch.setattr(storage, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(storage, "TEAM_RESULTS_DIR", team_dir)

    import project2_app.reporting as reporting

    monkeypatch.setattr(reporting, "create_result_directory", storage.create_result_directory)

    settings = load_config()
    dataset = load_dataset()
    split = split_dataset(dataset, settings)
    spec = build_model_spec("decision_tree", settings)
    training = train_model(spec, split, settings, "Standard")
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
    result_dir, payload = create_run_artifacts(
        member="Jesus Gaud",
        model_key="decision_tree",
        model_name="Decision Tree",
        training=training,
        evaluation=evaluation,
        settings=settings,
        feature_names=dataset.feature_names,
    )
    return result_dir, payload, evaluation, team_dir


def test_artifacts_save_package_import_and_compare(tmp_path: Path, monkeypatch) -> None:
    result_dir, payload, evaluation, team_dir = _completed_run(tmp_path, monkeypatch)

    expected = {
        "metrics.json",
        "manifest.json",
        "classification_report.csv",
        "predictions.csv",
        "roc_points.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "model_specific_plot.png",
        "run_summary.txt",
    }
    assert expected.issubset({p.name for p in result_dir.iterdir()})

    saved = save_team_result(result_dir, "Jesus Gaud")
    assert (saved / "metrics.json").exists()
    loaded = load_team_results()
    assert len(loaded) == 1
    assert loaded[0]["model_name"] == "Decision Tree"

    frame = team_comparison_frame(loaded)
    assert frame.iloc[0]["Member"] == "Jesus Gaud"
    assert frame.iloc[0]["Model"] == "Decision Tree"

    package_path = package_result(result_dir)
    assert package_path.exists()

    for child in team_dir.iterdir():
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
    destination, imported = import_result_package(package_path.read_bytes())
    assert destination.name == "jesus_gaud"
    assert imported["run_id"] == payload["run_id"]
    assert load_team_results()[0]["experiment_fingerprint"] == payload["experiment_fingerprint"]

    explanation = confusion_explanation(evaluation.confusion)
    assert "malignant" in explanation.lower()
    table = metric_table(evaluation.metrics)
    assert set(table["Metric"]) >= {"Accuracy", "Recall (Malignant)"}


def test_load_team_results_ignores_invalid_json(tmp_path: Path, monkeypatch) -> None:
    team_dir = tmp_path / "team_results"
    bad = team_dir / "bad_member"
    bad.mkdir(parents=True)
    (bad / "metrics.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(storage, "TEAM_RESULTS_DIR", team_dir)
    assert load_team_results() == []


def test_import_rejects_missing_required_files(tmp_path: Path, monkeypatch) -> None:
    import io
    import zipfile

    team_dir = tmp_path / "team_results"
    team_dir.mkdir()
    monkeypatch.setattr(storage, "TEAM_RESULTS_DIR", team_dir)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metrics.json", json.dumps({"member": "Someone"}))

    try:
        import_result_package(buffer.getvalue())
    except ValueError as exc:
        assert "manifest.json" in str(exc)
    else:
        raise AssertionError("Expected invalid package to be rejected")


def test_best_metric_summary_reports_display_precision_ties() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"Model": "Support Vector Machine", "Accuracy": 0.9736842},
            {"Model": "Logistic Regression", "Accuracy": 0.9736001},
            {"Model": "Random Forest", "Accuracy": 0.9649123},
        ]
    )

    best_value, winners = best_metric_summary(frame, "Accuracy", precision=3)

    assert best_value == 0.974
    assert winners == ["Logistic Regression", "Support Vector Machine"]


def test_best_metric_summary_reports_single_winner() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"Model": "Logistic Regression", "Recall": 0.952},
            {"Model": "Support Vector Machine", "Recall": 0.929},
        ]
    )

    best_value, winners = best_metric_summary(frame, "Recall", precision=3)

    assert best_value == 0.952
    assert winners == ["Logistic Regression"]
