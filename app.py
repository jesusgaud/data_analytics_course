from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from project2_app.config import ROOT_DIR, load_config
from project2_app.data import load_dataset, split_dataset
from project2_app.evaluation import confusion_explanation, evaluate_model
from project2_app.fingerprint import experiment_fingerprint
from project2_app.models import available_models, build_model_spec
from project2_app.reporting import (
    best_metric_summary,
    create_run_artifacts,
    team_comparison_frame,
)
from project2_app.storage import (
    TEAM_RESULTS_DIR,
    import_result_package,
    inspect_result_package,
    load_team_results,
    package_result,
    save_team_result,
    slugify,
)
from project2_app.training import train_model

st.set_page_config(
    page_title="IS 665 Project 2 Model Lab",
    page_icon="📊",
    layout="wide",
)

settings = load_config()
dataset = load_dataset()
models = available_models(settings)


def _result_dir_from_state() -> Path | None:
    value = st.session_state.get("result_dir")
    return Path(value) if value else None


def _metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(5)
    cards = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall (Malignant)", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]
    for column, (label, key) in zip(columns, cards, strict=True):
        column.metric(label, f"{metrics[key]:.3f}")


st.title("IS 665 Project 2 — Model Comparison Lab")
st.caption(
    "One controlled experiment, multiple supervised-learning algorithms, standardized results. "
    "No code editing required."
)

with st.sidebar:
    st.subheader("Experiment Contract")
    st.write(f"**Dataset:** {settings.dataset_name}")
    st.write(f"**Observations:** {len(dataset.X)}")
    st.write(f"**Predictors:** {dataset.X.shape[1]}")
    st.write(f"**Test split:** {settings.test_size:.0%}")
    st.write(f"**Random state:** {settings.random_state}")
    st.write(f"**Positive class:** {settings.positive_class.title()}")
    st.write(f"**Fingerprint:** `{experiment_fingerprint(settings)}`")
    st.info("These settings are locked so every teammate's model is evaluated under the same conditions.")

run_tab, results_tab, team_tab, method_tab = st.tabs(
    ["Run Analysis", "Individual Results", "Team Comparison", "Methodology"]
)

with run_tab:
    st.header("Run an Analysis")
    left, right = st.columns(2)
    member = left.selectbox("Team member", settings.team_members)
    display_to_key = {display: key for key, display in models.items()}
    model_display = right.selectbox("Model", list(display_to_key))
    model_key = display_to_key[model_display]

    existing = load_team_results()
    claimed_models = {
        str(item["model_name"]): str(item["member"])
        for item in existing
        if str(item["member"]) != member
    }
    if model_display in claimed_models:
        st.warning(
            f"{model_display} already has a saved team result from {claimed_models[model_display]}. "
            "You can still run it, but the team may prefer different algorithms for the final comparison."
        )

    mode = st.radio(
        "Run mode",
        options=["Standard", "Tuned"],
        horizontal=True,
        help="Standard uses the configured model settings. Tuned runs a modest 5-fold grid search on the training set.",
    )

    with st.expander("What the application will do"):
        st.markdown(
            """
            - Load the Breast Cancer Wisconsin Diagnostic dataset.
            - Use the same stratified 80/20 train/test split for every model.
            - Keep the test set untouched during model development.
            - Apply scaling only when the selected algorithm requires it.
            - Run 5-fold cross-validation on the training set.
            - Evaluate the final model on the fixed holdout test set.
            - Generate accuracy, precision, recall, F1, ROC-AUC, confusion matrix, predictions, and report artifacts.
            """
        )

    if st.button("Run Analysis", type="primary", width="stretch"):
        with st.spinner("Running the controlled experiment..."):
            split = split_dataset(dataset, settings)
            spec = build_model_spec(model_key, settings)
            training = train_model(spec, split, settings, mode)
            evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
            result_dir, metrics_payload = create_run_artifacts(
                member=member,
                model_key=model_key,
                model_name=model_display,
                training=training,
                evaluation=evaluation,
                settings=settings,
                feature_names=dataset.feature_names,
            )
            st.session_state["result_dir"] = str(result_dir)
            st.session_state["metrics_payload"] = metrics_payload
            st.session_state["confusion"] = evaluation.confusion.tolist()
            st.session_state["explanation"] = confusion_explanation(evaluation.confusion)
        st.success("Analysis complete. Open the Individual Results tab to review and save/export it.")

with results_tab:
    st.header("Individual Results")
    result_dir = _result_dir_from_state()
    payload = st.session_state.get("metrics_payload")

    if result_dir is None or payload is None or not result_dir.exists():
        st.info("Run an analysis first. The results for your current session will appear here.")
    else:
        st.subheader(f"{payload['model_name']} — {payload['member']}")
        _metric_cards(payload["metrics"])
        st.write(
            f"**5-fold CV mean F1:** {payload['cv_mean_f1']:.3f}  |  "
            f"**Run mode:** {payload['mode']}"
        )

        left, right = st.columns(2)
        confusion_path = result_dir / "confusion_matrix.png"
        roc_path = result_dir / "roc_curve.png"
        if confusion_path.exists():
            left.image(str(confusion_path), caption="Confusion Matrix", width="stretch")
        if roc_path.exists():
            right.image(str(roc_path), caption="ROC Curve", width="stretch")

        model_plot_path = result_dir / "model_specific_plot.png"
        if model_plot_path.exists():
            st.image(str(model_plot_path), caption="Model-Specific Interpretation", width="stretch")

        st.subheader("Interpretation")
        st.write(st.session_state.get("explanation", ""))
        st.info(
            "For this classification problem, recall for malignant cases deserves particular attention because "
            "a false negative is a malignant observation incorrectly classified as benign."
        )

        report_path = result_dir / "classification_report.csv"
        if report_path.exists():
            report = pd.read_csv(report_path, index_col=0)
            st.subheader("Classification Report")
            st.dataframe(report.round(3), width="stretch")

        st.subheader("Model Settings")
        st.json(payload.get("best_params", {}))

        action_left, action_middle, action_right = st.columns(3)
        if action_left.button("Save Team Results", width="stretch"):
            destination = save_team_result(result_dir, str(payload["member"]))
            st.success(f"Saved to the local team-results collection: {destination.name}")

        summary_path = result_dir / "run_summary.txt"
        if summary_path.exists():
            action_middle.download_button(
                "Download Run Summary",
                data=summary_path.read_bytes(),
                file_name=f"{payload['run_id']}_summary.txt",
                mime="text/plain",
                width="stretch",
            )

        package_path = package_result(result_dir)
        action_right.download_button(
            "Export Result Package",
            data=package_path.read_bytes(),
            file_name=package_path.name,
            mime="application/zip",
            width="stretch",
        )

with team_tab:
    st.header("Team Status & Comparison")
    st.write(
        "Each teammate runs locally. To combine results, either save results in a shared Git checkout or "
        "send the exported result ZIP to one teammate, who imports it here."
    )

    def _clear_pending_imports() -> None:
        st.session_state.pop("pending_import_batch", None)
        st.session_state.pop("pending_import_overwrites", None)

    def _package_label(uploaded_file: Any) -> str:
        return str(getattr(uploaded_file, "name", "uploaded_package.zip"))

    def _perform_import_batch(uploads: list[Any], *, allow_overwrite: bool) -> None:
        imported_messages: list[str] = []
        fingerprint_warnings: list[str] = []
        for uploaded_file in uploads:
            destination, imported = import_result_package(
                uploaded_file.getvalue(),
                allow_overwrite=allow_overwrite,
            )
            imported_messages.append(
                f"Imported {imported['member']} — {imported['model_name']} into {destination.name}."
            )
            if imported["experiment_fingerprint"] != experiment_fingerprint(settings):
                fingerprint_warnings.append(
                    f"{_package_label(uploaded_file)}: fingerprint mismatch for "
                    f"{imported['member']} — {imported['model_name']}."
                )

        if imported_messages:
            st.success("\n".join(imported_messages))
        if fingerprint_warnings:
            st.error(
                "One or more packages were imported, but their experiment fingerprint does not match this repository. "
                "Do not include those runs in the final comparison until the configuration is reconciled.\n\n"
                + "\n".join(fingerprint_warnings)
            )

    uploads = st.file_uploader(
        "Import a teammate result package",
        type=["zip"],
        accept_multiple_files=True,
        help=(
            "You can import one or more teammate result ZIP packages at the same time. "
            "Use the x next to any selected file if you want to remove it before importing."
        ),
    )

    selected_uploads = list(uploads) if uploads else []
    if selected_uploads and st.button("Import Result Package"):
        try:
            package_summaries: list[dict[str, str]] = []
            uploaded_members: dict[str, str] = {}
            overwrite_messages: list[str] = []

            for uploaded_file in selected_uploads:
                package_info = inspect_result_package(uploaded_file.getvalue())
                member = str(package_info["member"])
                model_name = str(package_info["model_name"])
                label = _package_label(uploaded_file)
                package_summaries.append(
                    {
                        "file_name": label,
                        "member": member,
                        "model_name": model_name,
                    }
                )

                if member in uploaded_members:
                    raise ValueError(
                        "You selected more than one package for the same teammate in this batch: "
                        f"{member}. Remove the duplicate package and try again."
                    )
                uploaded_members[member] = label

                destination = TEAM_RESULTS_DIR / slugify(member)
                existing_metrics_path = destination / "metrics.json"
                if existing_metrics_path.exists():
                    existing_metrics = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
                    overwrite_messages.append(
                        f"Existing result found: {member} currently has "
                        f"{existing_metrics['model_name']} saved. Importing {label} will replace it with "
                        f"{model_name}."
                    )

            if overwrite_messages:
                st.session_state["pending_import_batch"] = package_summaries
                st.session_state["pending_import_overwrites"] = overwrite_messages
            else:
                _perform_import_batch(selected_uploads, allow_overwrite=False)
                _clear_pending_imports()
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            _clear_pending_imports()
            st.error(f"Could not import package: {exc}")

    pending_batch = st.session_state.get("pending_import_batch")
    pending_overwrites = st.session_state.get("pending_import_overwrites")
    if pending_batch and pending_overwrites:
        st.warning(
            "Overwrite confirmation required. The following imported package(s) will replace existing saved team results:"
        )
        for message in pending_overwrites:
            st.write(f"- {message}")

        pending_frame = pd.DataFrame(
            [
                {
                    "Selected package": item["file_name"],
                    "Teammate": item["member"],
                    "New model": item["model_name"],
                }
                for item in pending_batch
            ]
        )
        st.dataframe(pending_frame, hide_index=True, width="stretch")

        replace_col, cancel_col = st.columns(2)
        if replace_col.button("Replace Existing Result(s)", width="stretch"):
            current_uploads = list(uploads) if uploads else []
            expected_names = [str(item["file_name"]) for item in pending_batch]
            current_names = [_package_label(uploaded_file) for uploaded_file in current_uploads]
            if current_names != expected_names:
                _clear_pending_imports()
                st.error(
                    "The selected upload list changed after the overwrite warning was shown. "
                    "Please click Import Result Package again to refresh the confirmation."
                )
            else:
                try:
                    _perform_import_batch(current_uploads, allow_overwrite=True)
                    _clear_pending_imports()
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    _clear_pending_imports()
                    st.error(f"Could not import package: {exc}")

        if cancel_col.button("Cancel", width="stretch"):
            _clear_pending_imports()
            st.info("Import cancelled. No existing team results were replaced.")

    team_results = load_team_results()
    result_by_member = {str(item["member"]): item for item in team_results}
    status_rows = []
    for team_member in settings.team_members:
        item = result_by_member.get(team_member)
        status_rows.append(
            {
                "Team Member": team_member,
                "Status": "Complete" if item else "Pending",
                "Model": item["model_name"] if item else "Not selected",
                "Mode": item["mode"] if item else "—",
            }
        )
    st.subheader("Team Status")
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

    if team_results:
        fingerprints = {str(item["experiment_fingerprint"]) for item in team_results}
        if len(fingerprints) > 1 or fingerprints != {experiment_fingerprint(settings)}:
            st.error(
                "Comparison blocked: one or more saved runs were produced under a different experiment contract. "
                "Re-run those models with the current repository configuration."
            )
        else:
            frame = team_comparison_frame(team_results)
            st.subheader("Model Comparison")
            comparison_for_display = frame.drop(columns=["Fingerprint"])
            st.dataframe(
                comparison_for_display.style.format(
                    {
                        "Accuracy": "{:.3f}",
                        "Precision": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1": "{:.3f}",
                        "ROC-AUC": "{:.3f}",
                        "CV Mean F1": "{:.3f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download Team Comparison CSV",
                data=comparison_for_display.to_csv(index=False).encode("utf-8"),
                file_name="is665_project2_team_model_comparison.csv",
                mime="text/csv",
            )

            metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            chart_frame = frame.set_index("Model")[metric_columns]
            st.bar_chart(chart_frame, stack=False, width="stretch")

            st.caption(
                "Best results are based on the team runs currently saved or imported "
                "into this application."
            )
            st.subheader("Best Results")
            summary_columns = st.columns(4)
            for column, metric in zip(
                summary_columns, ["Accuracy", "Recall", "F1", "ROC-AUC"], strict=True
            ):
                best_value, winning_models = best_metric_summary(frame, metric, precision=3)
                winner_label = (
                    winning_models[0]
                    if len(winning_models) == 1
                    else f"Tie: {' & '.join(winning_models)}"
                )
                column.metric(
                    f"Best {metric}",
                    f"{best_value:.3f}",
                    delta=winner_label,
                    delta_color="off",
                )

            duplicate_models = frame[frame.duplicated("Model", keep=False)]
            if not duplicate_models.empty:
                st.warning(
                    "More than one teammate has saved the same model. That is allowed for experimentation, "
                    "but the final assignment may be stronger if each member owns a different algorithm."
                )
    else:
        st.info("No team results have been saved or imported yet.")

with method_tab:
    st.header("Methodology & Reproducibility")
    st.markdown(
        f"""
        **Research question:** Can supervised machine-learning algorithms accurately classify breast tumors as
        malignant or benign, and which algorithm performs best under the same experimental conditions?

        **Dataset:** {settings.dataset_name}  
        **Source:** {settings.dataset_source}  
        **Observations:** {len(dataset.X):,}  
        **Predictors:** {dataset.X.shape[1]} numeric diagnostic features  
        **Positive class:** Malignant  
        **Holdout test size:** {settings.test_size:.0%}  
        **Random state:** {settings.random_state}  
        **Cross-validation:** {settings.cv_folds}-fold stratified CV on the training data  
        **Primary comparison metrics:** Accuracy, Precision, Recall, F1  
        **Supplemental metric:** ROC-AUC  
        **Experiment fingerprint:** `{experiment_fingerprint(settings)}`

        Preprocessing is encapsulated inside scikit-learn pipelines. Scaling is fitted on training folds only for
        algorithms that require it, which prevents test-set leakage. The fixed stratified holdout split keeps the
        malignant/benign class proportions comparable across all models.
        """
    )
    st.link_button("Open official UCI dataset page", settings.dataset_url)
    st.subheader("Configured Models")
    st.dataframe(
        pd.DataFrame(
            [{"Model Key": key, "Model": display} for key, display in models.items()]
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Locked Experiment Configuration")
    st.code((ROOT_DIR / "configs" / "experiment.toml").read_text(encoding="utf-8"), language="toml")
