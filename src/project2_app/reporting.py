from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from project2_app.config import ExperimentSettings
from project2_app.evaluation import EvaluationResult, confusion_explanation
from project2_app.fingerprint import experiment_contract, experiment_fingerprint
from project2_app.plots import save_confusion_matrix, save_model_specific_plot, save_roc_curve
from project2_app.storage import create_result_directory, slugify, write_json
from project2_app.training import TrainingResult


def _run_id(member: str, model_key: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify(member)}_{model_key}_{timestamp}"


def create_run_artifacts(
    member: str,
    model_key: str,
    model_name: str,
    training: TrainingResult,
    evaluation: EvaluationResult,
    settings: ExperimentSettings,
    feature_names: tuple[str, ...],
) -> tuple[Path, dict[str, Any]]:
    run_id = _run_id(member, model_key)
    result_dir = create_result_directory(run_id)
    created_at = datetime.now(UTC).isoformat()

    metrics_payload: dict[str, Any] = {
        "schema_version": settings.schema_version,
        "run_id": run_id,
        "created_at_utc": created_at,
        "member": member,
        "model_key": model_key,
        "model_name": model_name,
        "mode": training.mode,
        "experiment_fingerprint": experiment_fingerprint(settings),
        "cv_mean_f1": training.cv_mean,
        "cv_std_f1": training.cv_std,
        "metrics": evaluation.metrics,
        "best_params": training.best_params,
    }
    write_json(result_dir / "metrics.json", metrics_payload)

    evaluation.report.to_csv(result_dir / "classification_report.csv")
    evaluation.predictions.to_csv(result_dir / "predictions.csv", index_label="row_index")
    evaluation.roc_points.to_csv(result_dir / "roc_points.csv", index=False)

    save_confusion_matrix(evaluation.confusion, result_dir / "confusion_matrix.png")
    save_roc_curve(evaluation.roc_points, evaluation.metrics["roc_auc"], result_dir / "roc_curve.png")
    model_plot = save_model_specific_plot(
        training.estimator,
        feature_names,
        model_key,
        result_dir / "model_specific_plot.png",
    )
    if model_plot is None:
        candidate = result_dir / "model_specific_plot.png"
        if candidate.exists():
            candidate.unlink()

    manifest = {
        "run": {
            "run_id": run_id,
            "created_at_utc": created_at,
            "member": member,
            "model": model_name,
            "model_key": model_key,
            "mode": training.mode,
        },
        "experiment": experiment_contract(settings),
        "experiment_fingerprint": experiment_fingerprint(settings),
        "best_params": training.best_params,
        "artifacts": {
            "metrics": "metrics.json",
            "classification_report": "classification_report.csv",
            "predictions": "predictions.csv",
            "confusion_matrix": "confusion_matrix.png",
            "roc_curve": "roc_curve.png",
            "model_specific_plot": "model_specific_plot.png" if model_plot else None,
        },
    }
    write_json(result_dir / "manifest.json", manifest)

    summary = (
        f"{model_name} - {member}\n"
        f"Mode: {training.mode}\n"
        f"5-fold CV mean F1: {training.cv_mean:.3f}\n"
        f"Accuracy: {evaluation.metrics['accuracy']:.3f}\n"
        f"Precision: {evaluation.metrics['precision']:.3f}\n"
        f"Recall (malignant): {evaluation.metrics['recall']:.3f}\n"
        f"F1: {evaluation.metrics['f1']:.3f}\n"
        f"ROC-AUC: {evaluation.metrics['roc_auc']:.3f}\n\n"
        f"{confusion_explanation(evaluation.confusion)}\n"
    )
    (result_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    return result_dir, metrics_payload


def best_metric_summary(
    frame: pd.DataFrame, metric: str, *, precision: int = 3
) -> tuple[float, list[str]]:
    """Return the displayed best metric value and every model tied at that precision."""
    if frame.empty:
        raise ValueError("Cannot summarize best metric from an empty comparison frame.")
    if metric not in frame.columns:
        raise KeyError(f"Metric column not found: {metric}")
    if "Model" not in frame.columns:
        raise KeyError("Model column not found in comparison frame.")

    displayed_values = frame[metric].astype(float).round(precision)
    best_value = float(displayed_values.max())
    winners = sorted(
        frame.loc[displayed_values.eq(best_value), "Model"].astype(str).unique().tolist()
    )
    return best_value, winners


def team_comparison_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        metrics = item["metrics"]
        rows.append(
            {
                "Member": item["member"],
                "Model": item["model_name"],
                "Mode": item["mode"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
                "CV Mean F1": item["cv_mean_f1"],
                "Fingerprint": item["experiment_fingerprint"],
            }
        )
    return pd.DataFrame(rows)
