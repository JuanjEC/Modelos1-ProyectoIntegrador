"""Estimadores y metricas."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ALGORITHMS = ("random_forest", "ridge")


def make_estimator(algorithm: str, seed: int, n_estimators: int = 120, max_depth: int = 20,
                   min_samples_leaf: int = 2, alpha: float = 10.0, log_target: bool = False):
    """Devuelve (estimador, parametros_principales)."""
    if algorithm == "random_forest":
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": seed,
        }
        estimator = RandomForestRegressor(n_jobs=-1, **params)
    elif algorithm == "ridge":
        params = {"alpha": alpha}
        estimator = Ridge(**params)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}. Opciones: {ALGORITHMS}")

    if log_target:
        # Entrena sobre log(1+Price) y devuelve la prediccion en AED (nunca negativa).
        estimator = TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )
    params = {**params, "log_target": bool(log_target)}
    return estimator, params


def make_baseline():
    """Modelo base: predice siempre la mediana del precio de entrenamiento."""
    return DummyRegressor(strategy="median")


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
