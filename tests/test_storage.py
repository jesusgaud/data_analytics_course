import io
import json
import zipfile
from pathlib import Path

from project2_app.storage import slugify


def test_slugify() -> None:
    assert slugify("Blessy Ann Thomas") == "blessy_ann_thomas"
    assert slugify("Support Vector Machine") == "support_vector_machine"


def test_result_package_shape_can_be_created(tmp_path: Path) -> None:
    metrics = {
        "member": "Jesus Gaud",
        "model_key": "random_forest",
        "model_name": "Random Forest",
        "experiment_fingerprint": "1234567890abcdef",
        "metrics": {"accuracy": 0.95},
        "run_id": "example_run",
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (source / "manifest.json").write_text(json.dumps({"run": "example"}), encoding="utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(source / "metrics.json", "metrics.json")
        archive.write(source / "manifest.json", "manifest.json")

    with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
        assert set(archive.namelist()) == {"metrics.json", "manifest.json"}
