# Contributing

## Branching

Use short-lived feature branches from `main`, for example:

- `feature/team-comparison`
- `fix/result-import`
- `docs/report-methodology`

Do not develop directly on `main` once branch protection is enabled.

## Local quality gate

Before opening a pull request on Windows, run `RUN_TESTS.bat` or:

```text
ruff check .
mypy src
pytest --cov=project2_app --cov-report=term-missing --cov-fail-under=75
```

GitHub Actions repeats the quality gate across Python 3.11 through 3.15.

## Experiment controls

Do not change the dataset, holdout split, random state, positive-class definition, cross-validation contract, or evaluation contract without team agreement. These values define the experiment fingerprint used to validate comparable runs.

## Generated artifacts

`results/` and `team_results/` are intentionally excluded from Git except for placeholder files. Share teammate results through the application export/import workflow unless the team deliberately decides to version final artifacts.
