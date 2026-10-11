"""Pruebas basicas (pytest). Usan datos sinteticos pequenos y directorios temporales (tmp_path)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS, PREDICTION_COLUMN  # noqa: E402
from src.errors import DataValidationError, VersionExistsError  # noqa: E402
from src.prediction import predict_dataframe, run_prediction  # noqa: E402
from src.training import run_training  # noqa: E402
from src.versioning import load_model  # noqa: E402

FAST = dict(n_estimators=5, max_depth=4, max_tfidf_features=20)


def make_df(n=120, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    makes = {"toyota": ["camry", "rush"], "bmw": ["x5", "m3"], "kia": ["rio", "sportage"]}
    rows = []
    for i in range(n):
        make = list(makes)[i % 3]
        model = makes[make][(i // 3) % 2]
        year = int(rng.integers(2005, 2025))
        km = int(rng.integers(10_000, 300_000))
        row = {
            "Make": make, "Model": model, "Year": year, "Mileage": km,
            "Body Type": "Sedan" if i % 2 else "SUV",
            "Cylinders": str(int(rng.choice([4, 6, 8]))),
            "Transmission": "Automatic Transmission",
            "Fuel Type": "Gasoline",
            "Color": ["Black", "White"][i % 2],
            "Location": " Dubai" if i % 4 == 0 else "Abu Dhabi",
            "Description": f"{year} {make} {model} with Bluetooth, Sunroof. Condition: No damage. #{i}",
        }
        row["Price"] = 20_000 + (year - 2005) * 3_000 - km * 0.05 + (30_000 if make == "bmw" else 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df if with_target else df.drop(columns=["Price"])


@pytest.fixture()
def data_csv(tmp_path):
    path = tmp_path / "datos.csv"
    make_df().to_csv(path, index=False)
    return path


@pytest.fixture()
def trained(tmp_path, data_csv):
    out = tmp_path / "models"
    res = run_training(data_csv, out, "v1", seed=1, **FAST)
    return out, res


# 1. Entrenamiento y almacenamiento
def test_train_saves_loadable_model_metrics_and_metadata(trained):
    out, res = trained
    vdir = out / "v1"
    assert (vdir / "modelo.joblib").is_file()
    metrics = json.loads((vdir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
    assert {"MAE", "RMSE", "R2"} <= set(metrics["modelo"])
    assert {"MAE", "RMSE", "R2"} <= set(metrics["baseline_mediana"])
    for key in ("version", "created_at", "target", "algorithm", "seed", "input_features",
                "split", "data", "python_version", "git_commit"):
        assert key in meta
    assert meta["version"] == "v1" and meta["target"] == "Price"
    assert meta["split"]["n_train"] + meta["split"]["n_test"] == meta["data"]["n_rows_used"]
    assert load_model(vdir) is not None


# 2. Carga y prediccion
def test_loaded_model_gives_one_prediction_per_row_and_matches_in_memory(trained, tmp_path):
    out, _ = trained
    new = make_df(15, seed=5, with_target=False)
    in_path, out_path = tmp_path / "in.csv", tmp_path / "pred.csv"
    new.to_csv(in_path, index=False)

    run_prediction(out / "v1" / "modelo.joblib", in_path, out_path)
    result = pd.read_csv(out_path)
    expected = joblib.load(out / "v1" / "modelo.joblib").predict(new)

    assert len(result) == len(new)
    assert PREDICTION_COLUMN in result.columns
    assert np.allclose(result[PREDICTION_COLUMN].to_numpy(), expected, atol=0.01)
    assert list(result["Make"]) == list(new["Make"])  # conserva el orden


# 3. Valores faltantes
def test_missing_values_are_handled_by_saved_strategy(trained):
    out, _ = trained
    model = load_model(out / "v1")
    new = make_df(6, seed=9, with_target=False)
    new.loc[0, "Cylinders"] = np.nan
    new.loc[1, "Cylinders"] = "Unknown"
    new.loc[2, "Mileage"] = np.nan
    new.loc[3, "Color"] = np.nan
    new.loc[4, "Description"] = np.nan
    result = predict_dataframe(model, new)
    assert len(result) == 6
    assert result[PREDICTION_COLUMN].notna().all()
    assert np.isfinite(result[PREDICTION_COLUMN]).all()


# 4. Entrada invalida
def test_missing_required_column_raises_clear_error(trained):
    out, _ = trained
    model = load_model(out / "v1")
    new = make_df(4, seed=2, with_target=False).drop(columns=["Mileage"])
    with pytest.raises(DataValidationError, match="Mileage"):
        predict_dataframe(model, new)


def test_cli_invalid_input_exits_nonzero_and_writes_no_output(trained, tmp_path):
    out, _ = trained
    bad = tmp_path / "bad.csv"
    make_df(4, seed=2, with_target=False).drop(columns=["Year"]).to_csv(bad, index=False)
    out_csv = tmp_path / "salida.csv"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "predict.py"), "--model", str(out / "v1"),
         "--input", str(bad), "--output", str(out_csv)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "Year" in proc.stderr
    assert not out_csv.exists()


def test_non_numeric_year_raises(trained):
    out, _ = trained
    new = make_df(4, seed=2, with_target=False)
    new["Year"] = new["Year"].astype(object)
    new.loc[1, "Year"] = "dos mil"
    with pytest.raises(DataValidationError, match="Year"):
        predict_dataframe(load_model(out / "v1"), new)


# 5. Versionamiento
def test_second_version_keeps_first_and_duplicate_id_is_rejected(trained, data_csv):
    out, _ = trained
    first = out / "v1" / "modelo.joblib"
    digest = hashlib.sha256(first.read_bytes()).hexdigest()

    run_training(data_csv, out, "v2", seed=2, log_target=True, **FAST)
    assert load_model(out / "v1") is not None
    assert load_model(out / "v2") is not None
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest

    with pytest.raises(VersionExistsError):
        run_training(data_csv, out, "v1", seed=3, **FAST)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest


# Extra: categorias nuevas, columnas extra y orden de columnas
def test_unseen_category_extra_and_reordered_columns(trained):
    out, _ = trained
    model = load_model(out / "v1")
    new = make_df(5, seed=4, with_target=False)
    new.loc[0, "Make"] = "marca-nueva"
    new["columna_extra"] = 1
    new = new[list(reversed(new.columns))]
    result = predict_dataframe(model, new)
    assert len(result) == 5 and "columna_extra" in result.columns
    assert result[PREDICTION_COLUMN].notna().all()
    assert set(FEATURE_COLUMNS) <= set(result.columns)
