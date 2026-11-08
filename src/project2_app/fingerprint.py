from __future__ import annotations

import hashlib
import json
from typing import Any

from project2_app.config import ExperimentSettings


def experiment_contract(settings: ExperimentSettings) -> dict[str, Any]:
    return {
        "dataset": settings.dataset_name,
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "positive_class": settings.positive_class,
        "cv_folds": settings.cv_folds,
        "scoring": settings.scoring,
        "schema_version": settings.schema_version,
    }


def experiment_fingerprint(settings: ExperimentSettings) -> str:
    payload = json.dumps(experiment_contract(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
