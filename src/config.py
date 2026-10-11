"""Esquema del problema: prediccion del precio (AED) de vehiculos usados en UAE."""

DEFAULT_TARGET = "Price"

NUMERIC_FEATURES = ["Year", "Mileage", "Cylinders"]
CATEGORICAL_FEATURES = [
    "Make",
    "Model",
    "Body Type",
    "Transmission",
    "Fuel Type",
    "Color",
    "Location",
]
TEXT_FEATURE = "Description"

# Orden canonico de las variables de entrada (el CSV puede traerlas en cualquier orden).
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

# Year y Mileage deben ser numericas. Cylinders es tolerante: textos como "Unknown" pasan a nulo.
STRICT_NUMERIC = ["Year", "Mileage"]

MODEL_FILENAME = "modelo.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"
PREDICTION_COLUMN = "prediccion"
