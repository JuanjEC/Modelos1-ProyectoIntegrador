"""Flujo completo de entrenamiento (reutilizable desde train.py, pruebas y, mas adelante, la API)."""

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_FEATURES, DEFAULT_TARGET, NUMERIC_FEATURES, TEXT_FEATURE
from .data import prepare_training_data, read_csv
from .modeling import make_baseline, make_estimator, regression_metrics
from .preprocessing import build_pipeline
from .versioning import create_version_dir, save_version, validate_version_id

DATA_SOURCE = "https://www.kaggle.com/datasets/mohamedsaad254/uae-used-cars-analysis-full-project-v1-0"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Hash del commit actual; en Docker (sin .git) se puede pasar con la variable GIT_COMMIT."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("GIT_COMMIT", "unknown")


def run_training(data_path, output_dir, version, target=DEFAULT_TARGET, seed=42, test_size=0.2,
                 algorithm="random_forest", log_target=False, n_estimators=120, max_depth=20,
                 min_samples_leaf=2, alpha=10.0, max_tfidf_features=300) -> dict:
    """Entrena, evalua (modelo y baseline) y guarda una version nueva. Devuelve rutas y metricas."""
    validate_version_id(version)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0 y 1 (sin incluirlos).")

    df = read_csv(data_path)
    X, y, data_summary = prepare_training_data(df, target)

    # 1) Separacion ANTES de ajustar cualquier transformacion que aprenda de los datos.
    #    No hay identificadores de grupo (cada fila es un anuncio distinto y no hay filas repetidas
    #    en las variables predictoras), por eso se usa una separacion aleatoria.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 2) Modelo base y modelo final: misma particion y misma metrica.
    baseline = build_pipeline(make_baseline(), max_tfidf_features)
    baseline.fit(X_train, y_train)
    baseline_metrics = regression_metrics(y_test, baseline.predict(X_test))

    estimator, params = make_estimator(
        algorithm, seed, n_estimators, max_depth, min_samples_leaf, alpha, log_target
    )
    model = build_pipeline(estimator, max_tfidf_features)
    model.fit(X_train, y_train)  # imputacion/escalado/codificacion/TF-IDF aprenden solo de X_train
    model_metrics = regression_metrics(y_test, model.predict(X_test))

    improvement = {
        "MAE_reduccion_pct": 100.0 * (1 - model_metrics["MAE"] / baseline_metrics["MAE"]),
    }
    metrics = {
        "metrica_principal": "MAE",
        "modelo": model_metrics,
        "baseline_mediana": baseline_metrics,
        "mejora_vs_baseline": improvement,
        "evaluado_sobre": f"conjunto de prueba ({len(X_test)} filas)",
    }

    # 3) Guardado (solo si todo lo anterior termino bien; no sobrescribe versiones existentes).
    version_dir = create_version_dir(output_dir, version)
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem_type": "regression",
        "target": target,
        "algorithm": algorithm,
        "model_params": params,
        "max_tfidf_features": max_tfidf_features,
        "seed": seed,
        "input_features": {
            **{c: "numerico" for c in NUMERIC_FEATURES},
            **{c: "categorico (texto)" for c in CATEGORICAL_FEATURES},
            TEXT_FEATURE: "texto libre",
        },
        "data": {
            "source": DATA_SOURCE,
            "file": Path(data_path).name,
            "sha256": _sha256(data_path),
            **data_summary,
        },
        "split": {
            "strategy": "train_test_split aleatorio (sin grupos)",
            "test_size": test_size,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "evaluation_metric": "MAE (tambien RMSE y R2)",
        "python_version": platform.python_version(),
        "requirements_file": "requirements.txt",
        "git_commit": _git_commit(),
    }
    paths = save_version(version_dir, model, metrics, metadata)
    return {"paths": {k: str(v) for k, v in paths.items()}, "metrics": metrics, "metadata": metadata}
