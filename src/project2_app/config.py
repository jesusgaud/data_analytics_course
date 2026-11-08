from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "experiment.toml"


@dataclass(frozen=True)
class ExperimentSettings:
    project_name: str
    course: str
    dataset_name: str
    dataset_source: str
    dataset_url: str
    team_members: tuple[str, ...]
    test_size: float
    random_state: int
    positive_class: str
    cv_folds: int
    scoring: str
    schema_version: str
    models: dict[str, dict[str, Any]]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ExperimentSettings:
    """Load and validate application configuration from TOML."""
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    required_sections = {"project", "team", "experiment", "models"}
    missing = required_sections.difference(raw)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    members = tuple(raw["team"]["members"])
    if not members:
        raise ValueError("At least one team member must be configured.")

    test_size = float(raw["experiment"]["test_size"])
    if not 0 < test_size < 1:
        raise ValueError("experiment.test_size must be between 0 and 1.")

    return ExperimentSettings(
        project_name=str(raw["project"]["name"]),
        course=str(raw["project"]["course"]),
        dataset_name=str(raw["project"]["dataset_name"]),
        dataset_source=str(raw["project"]["dataset_source"]),
        dataset_url=str(raw["project"]["dataset_url"]),
        team_members=members,
        test_size=test_size,
        random_state=int(raw["experiment"]["random_state"]),
        positive_class=str(raw["experiment"]["positive_class"]),
        cv_folds=int(raw["experiment"]["cv_folds"]),
        scoring=str(raw["experiment"]["scoring"]),
        schema_version=str(raw["experiment"]["schema_version"]),
        models=dict(raw["models"]),
    )
