"""Prediccion sobre datos nuevos usando una version guardada (sin reentrenar ni reajustar nada)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, PREDICTION_COLUMN
from .data import read_csv, validate_columns, validate_types
from .errors import DataValidationError
from .versioning import load_model


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame original (mismo orden) mas la columna 'prediccion'.

    - Columnas faltantes -> DataValidationError. Columnas extra se ignoran (se conservan en la salida).
    - Year/Mileage no numericos -> DataValidationError.
    - Nulos: los trata la estrategia guardada en el modelo (mediana / moda / texto vacio).
    - Categorias no vistas en entrenamiento: se ignoran (codificacion en ceros).
    - Cada fila produce una prediccion; no se eliminan filas.
    """
    validate_columns(df, FEATURE_COLUMNS)
    validate_types(df)
    preds = np.asarray(model.predict(df), dtype=float)
    if len(preds) != len(df):
        raise DataValidationError("El modelo no genero una prediccion por cada fila de entrada.")
    out = df.copy()
    out[PREDICTION_COLUMN] = np.round(preds, 2)
    return out


def run_prediction(model_path, input_path, output_path) -> dict:
    """Carga el modelo, predice y escribe el CSV de salida SOLO si todo salio bien."""
    model = load_model(model_path)
    df = read_csv(input_path)
    result = predict_dataframe(model, df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(output_path)  # escritura atomica: nunca queda un archivo a medias
    return {"n_rows": int(len(result)), "output": str(output_path)}
