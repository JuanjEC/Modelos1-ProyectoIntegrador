"""Lectura, validacion y limpieza determinista de datos (sin aprender nada de los datos)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    STRICT_NUMERIC,
    TEXT_FEATURE,
)
from .errors import DataValidationError


def read_csv(path) -> pd.DataFrame:
    """Lee un CSV y devuelve un DataFrame, con errores comprensibles."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"No existe el archivo CSV: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas lanza varios tipos de error segun el problema
        raise DataValidationError(f"No se pudo leer el CSV {path}: {exc}") from exc
    if df.empty:
        raise DataValidationError(f"El archivo CSV esta vacio: {path}")
    return df


def validate_columns(df: pd.DataFrame, required) -> None:
    """Verifica que existan las columnas requeridas (una celda nula no cuenta como columna ausente)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Faltan columnas requeridas: {missing}. Columnas recibidas: {list(df.columns)}"
        )


def validate_types(df: pd.DataFrame) -> None:
    """Year y Mileage deben ser numericas (se admiten nulos). Cylinders se limpia despues."""
    for col in STRICT_NUMERIC:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & converted.isna()
        if bad.any():
            rows = [int(i) for i in df.index[bad][:5]]
            raise DataValidationError(
                f"La columna '{col}' debe ser numerica; hay {int(bad.sum())} valores no "
                f"numericos (primeras filas con problema, indice 0): {rows}"
            )


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza determinista de las variables de entrada.

    Se usa dentro del Pipeline, por lo que entrenamiento y prediccion comparten exactamente
    la misma logica. No aprende nada de los datos.
    - Selecciona y ordena las columnas segun FEATURE_COLUMNS (ignora columnas extra, como Price).
    - Variables numericas: conversion a numero; lo no convertible (p. ej. "Unknown") pasa a NaN.
    - Variables categoricas: quita espacios al inicio/fin (en el CSV hay " Dubai" y "Dubai").
    - Description: nulos -> cadena vacia.
    """
    validate_columns(df, FEATURE_COLUMNS)
    out = df[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    for col in CATEGORICAL_FEATURES:
        s = out[col].astype("object")
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: np.nan if (isinstance(v, str) and v == "") else v)
        out[col] = s.where(pd.notna(s), np.nan)

    desc = out[TEXT_FEATURE].astype("object")
    out[TEXT_FEATURE] = desc.where(pd.notna(desc), "").astype(str)
    return out


def prepare_training_data(df: pd.DataFrame, target: str):
    """Transformaciones deterministas previas a la separacion train/test.

    - Convierte el objetivo a numero y elimina filas cuyo objetivo no se puede usar.
    - Elimina duplicados exactos (para que no queden copias en train y test).
    Devuelve (X, y, resumen). La imputacion y demas transformaciones aprendidas NO se hacen aqui.
    """
    validate_columns(df, FEATURE_COLUMNS + [target])
    validate_types(df)

    n_raw = len(df)
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    n_after_target = len(work)
    work = work.drop_duplicates()
    n_after_dups = len(work)

    if n_after_dups < 20:
        raise DataValidationError(
            f"Quedan solo {n_after_dups} filas utilizables; se necesitan al menos 20 para entrenar."
        )

    X = work[FEATURE_COLUMNS].copy()
    y = work[target].astype(float)
    summary = {
        "n_rows_raw": int(n_raw),
        "n_dropped_invalid_target": int(n_raw - n_after_target),
        "n_dropped_duplicates": int(n_after_target - n_after_dups),
        "n_rows_used": int(n_after_dups),
    }
    return X, y, summary
