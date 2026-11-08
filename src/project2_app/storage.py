from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from project2_app.config import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "results"
TEAM_RESULTS_DIR = ROOT_DIR / "team_results"
REQUIRED_PACKAGE_FIELDS = {
    "member",
    "model_key",
    "model_name",
    "experiment_fingerprint",
    "metrics",
    "run_id",
}


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_result_directory(run_id: str) -> Path:
    path = RESULTS_DIR / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_team_result(result_dir: Path, member: str) -> Path:
    destination = TEAM_RESULTS_DIR / slugify(member)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(result_dir, destination)
    return destination


def package_result(result_dir: Path) -> Path:
    zip_path = result_dir.parent / f"{result_dir.name}_result_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in result_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(result_dir))
    return zip_path


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise ValueError("Result package contains an unsafe path.")
    archive.extractall(destination)


def _read_result_package(package_bytes: bytes) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_dir.name)
    package_path = temp_root / "package.zip"
    package_path.write_bytes(package_bytes)
    extract_dir = temp_root / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(package_path, "r") as archive:
        _safe_extract(archive, extract_dir)

    metrics_path = extract_dir / "metrics.json"
    manifest_path = extract_dir / "manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        temp_dir.cleanup()
        raise ValueError("Invalid result package: metrics.json and manifest.json are required.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    missing = REQUIRED_PACKAGE_FIELDS.difference(metrics)
    if missing:
        temp_dir.cleanup()
        raise ValueError(f"Invalid result package: missing fields {sorted(missing)}")

    return extract_dir, metrics, temp_dir


def inspect_result_package(package_bytes: bytes) -> dict[str, Any]:
    extract_dir, metrics, temp_dir = _read_result_package(package_bytes)
    try:
        return {
            "member": str(metrics["member"]),
            "model_key": str(metrics["model_key"]),
            "model_name": str(metrics["model_name"]),
            "experiment_fingerprint": str(metrics["experiment_fingerprint"]),
            "run_id": str(metrics["run_id"]),
            "destination": str(TEAM_RESULTS_DIR / slugify(str(metrics["member"]))),
        }
    finally:
        temp_dir.cleanup()


def import_result_package(
    package_bytes: bytes,
    *,
    allow_overwrite: bool = True,
) -> tuple[Path, dict[str, Any]]:
    extract_dir, metrics, temp_dir = _read_result_package(package_bytes)
    try:
        destination = TEAM_RESULTS_DIR / slugify(str(metrics["member"]))
        if destination.exists():
            if not allow_overwrite:
                raise ValueError(
                    f"A saved result already exists for {metrics['member']}. "
                    "Confirm replacement before importing this package."
                )
            shutil.rmtree(destination)
        shutil.copytree(extract_dir, destination)
        return destination, metrics
    finally:
        temp_dir.cleanup()


def load_team_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not TEAM_RESULTS_DIR.exists():
        return results
    for metrics_path in sorted(TEAM_RESULTS_DIR.glob("*/metrics.json")):
        try:
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return results
