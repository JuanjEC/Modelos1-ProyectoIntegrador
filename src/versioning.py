"""Versionamiento de modelos por carpetas: models/<version>/{modelo.joblib, metrics.json, metadata.json}."""

import json
import re
from pathlib import Path

import joblib

from .config import METADATA_FILENAME, METRICS_FILENAME, MODEL_FILENAME
from .errors import ModelLoadError, VersionExistsError

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_version_id(version: str) -> None:
    if not _VERSION_RE.match(version or ""):
        raise ValueError(
            f"Identificador de version invalido: {version!r}. Use letras, numeros, '.', '_' o '-'."
        )


def create_version_dir(output_dir, version: str) -> Path:
    """Crea la carpeta de la version. Rechaza identificadores repetidos (no sobrescribe)."""
    validate_version_id(version)
    version_dir = Path(output_dir) / version
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        version_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise VersionExistsError(
            f"La version '{version}' ya existe en {version_dir}. Elija otro identificador."
        ) from exc
    return version_dir


def save_version(version_dir: Path, model, metrics: dict, metadata: dict) -> dict:
    paths = {
        "model": version_dir / MODEL_FILENAME,
        "metrics": version_dir / METRICS_FILENAME,
        "metadata": version_dir / METADATA_FILENAME,
    }
    joblib.dump(model, paths["model"], compress=3)
    for key, content in (("metrics", metrics), ("metadata", metadata)):
        with open(paths[key], "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    return paths


def resolve_model_path(path) -> Path:
    """Acepta la ruta del .joblib o la carpeta de la version."""
    p = Path(path)
    if p.is_dir():
        p = p / MODEL_FILENAME
    if not p.is_file():
        raise ModelLoadError(f"No se encontro el modelo: {p}")
    return p


def load_model(path):
    p = resolve_model_path(path)
    try:
        return joblib.load(p)
    except Exception as exc:
        raise ModelLoadError(f"No se pudo cargar el modelo {p}: {exc}") from exc
