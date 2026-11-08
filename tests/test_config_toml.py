from pathlib import Path

import pytest

from project2_app.config import load_config


def test_toml_config_loads_without_third_party_parser() -> None:
    settings = load_config()
    assert settings.project_name.startswith("IS 665 Project 2")
    assert settings.models["random_forest"]["display_name"] == "Random Forest"


def test_invalid_toml_config_rejects_bad_test_size(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '''
[project]
name = "x"
course = "x"
dataset_name = "x"
dataset_source = "x"
dataset_url = "x"
[team]
members = ["A"]
[experiment]
test_size = 1.5
random_state = 42
positive_class = "malignant"
cv_folds = 5
scoring = "f1"
schema_version = "1"
[models]
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test_size"):
        load_config(path)
