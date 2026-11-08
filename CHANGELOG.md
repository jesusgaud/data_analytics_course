## v1.2.3 - import workflow refinement

- Added overwrite confirmation before an imported package replaces an existing saved teammate result.
- Added support for selecting and importing multiple result-package ZIP files in one step.
- Added batch validation that blocks duplicate teammate packages in the same import batch.
- Kept the built-in remove-file behavior so selected packages can be cleared with the x control before importing.

# Changelog

## 1.2.0 — TOML configuration and Python 3.15 hardening

- Removed PyYAML and `pyyaml-cp315` from all runtime/development dependencies.
- Replaced `configs/experiment.yaml` with `configs/experiment.toml`.
- Configuration loading now uses Python 3.11+ standard-library `tomllib`.
- Replaced exported `manifest.yaml` files with `manifest.json`.
- Updated result-package validation and automated tests for JSON manifests.
- Added TOML configuration tests.
- Pinned pandas 3.1.1, NumPy 2.5.4, and Matplotlib 3.11.2 for reproducible Python 3.15 Windows installs.
- Revalidated all five standard models and the tuned Random Forest path.
- Regression suite increased to 17 tests with 92.72% package coverage.
- Updated documentation and release examples to version 1.2.0.

## 1.1.0 — Python 3.15 compatibility review

- Expanded supported Python range through Python 3.15.
- Updated Streamlit to 1.66.1.
- Updated pandas dependency range for Python 3.15 compatibility.
- Expanded CI matrix to Python 3.11–3.15.
- Moved smoke/release jobs to Python 3.15.
- Improved Windows launcher runtime detection and packaging-tool upgrades.
- Added `.python-version` with Python 3.15 as the preferred development runtime.
- Added downloadable individual run summaries.
- Added downloadable team-comparison CSV.
- Added complete Windows/VS Code/GitHub setup documentation.
- Refreshed validation and collaboration documentation.
