# Validation Record

Validation performed for repository version **1.2.0** after removing the YAML/PyYAML dependency and completing the Python 3.15 compatibility review.

## Review scope

- Python/runtime metadata
- runtime and development dependencies
- TOML experiment configuration via Python standard-library `tomllib`
- JSON result manifests
- Windows launcher behavior
- Streamlit source and imports
- model factory and training pipeline
- evaluation/reporting
- result package export/import
- experiment fingerprint controls
- automated tests and coverage
- GitHub Actions CI matrix
- release/CD workflow
- documentation and team workflow

## Local validation completed in the packaging environment

- Python source compilation: **passed** for `app.py`, `src/`, and `tests/`.
- Pytest suite: **17 tests passed**.
- Core package test coverage: **92.72%**.
- Standard end-to-end execution passed for all five supported models.
- Tuned Random Forest execution passed.
- Result artifact creation, ZIP packaging/import, team storage, and comparison logic remain covered by automated tests.
- Repository scan confirms no application/test dependency on PyYAML, `pyyaml-cp315`, `import yaml`, `experiment.yaml`, or `manifest.yaml`.

### Standard-run regression values

| Model | Accuracy | Recall (Malignant) | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.974 | 0.952 | 0.964 | 0.996 |
| Decision Tree | 0.921 | 0.833 | 0.886 | 0.945 |
| Random Forest | 0.965 | 0.905 | 0.950 | 0.994 |
| Support Vector Machine | 0.974 | 0.929 | 0.963 | 0.995 |
| K-Nearest Neighbors | 0.956 | 0.905 | 0.938 | 0.982 |

## Python 3.15 dependency review

The project metadata supports Python **3.11 through 3.15**, with Python 3.15 preferred for new development. Runtime dependencies are pinned to versions reviewed for Python 3.15 support: Streamlit 1.66.1, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4, and Matplotlib 3.11.2.

The application configuration is now parsed by `tomllib`, which is part of Python 3.11+, so no third-party YAML parser or C/C++ compiler is required for configuration handling.

## Packaging-environment limitation

The packaging container cannot install a fresh dependency set from PyPI, so Ruff, mypy, and a newly installed Streamlit runtime could not be executed here. The analytical regression suite, source compilation, standard model executions, tuned Random Forest execution, and configuration/result-package paths were executed against the available environment. The included GitHub Actions workflow is the authoritative clean-environment gate: it installs the declared dependencies, runs Ruff, mypy, pytest/coverage across Python 3.11–3.15, and performs the Streamlit/import smoke test on Python 3.15.
