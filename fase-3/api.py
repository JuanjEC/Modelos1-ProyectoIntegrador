import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS, PREDICTION_COLUMN
from src.errors import DataValidationError, ModelLoadError
from src.monitoring import append_event
from src.prediction import predict_dataframe
from src.training import run_training
from src.versioning import load_model, validate_version_id


def validate_record(record):
    if not isinstance(record, dict):
        raise HTTPException(422, "Se requiere un objeto JSON con las variables del vehiculo.")
    missing = [name for name in FEATURE_COLUMNS if name not in record]
    if missing:
        raise HTTPException(422, f"Faltan campos obligatorios: {', '.join(missing)}")
    for name in FEATURE_COLUMNS:
        value = record[name]
        if value is None:
            continue
        if name in ("Year", "Mileage"):
            valid = type(value) in (int, float)
        elif name == "Cylinders":
            valid = type(value) in (int, float, str)
        else:
            valid = isinstance(value, str)
        if not valid or (type(value) in (int, float) and not math.isfinite(value)):
            expected = "numero o null" if name in ("Year", "Mileage") else "texto o null"
            if name == "Cylinders":
                expected = "numero, texto o null"
            raise HTTPException(422, f"El campo '{name}' debe ser {expected}.")
    return {name: record[name] for name in FEATURE_COLUMNS}


def create_app(*, models_dir=None, model_version=None, data_path=None, log_path=None,
               training_options=None):
    models_dir = Path(models_dir or os.getenv("MODELS_DIR", ROOT / "models"))
    model_version = model_version or os.getenv("MODEL_VERSION", "v2")
    data_path = Path(data_path or os.getenv("TRAIN_DATA", ROOT / "data/uae_used_cars_10k.csv"))
    log_path = Path(log_path or os.getenv("PREDICTION_LOG", ROOT / "outputs/solicitudes.jsonl"))
    validate_version_id(model_version)
    application = FastAPI(title="Precios de vehiculos usados UAE", version="3.0")

    @application.middleware("http")
    async def record_prediction(request: Request, call_next):
        tracked = request.method == "POST" and request.url.path.rstrip("/") == "/predict"
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(status_code=500, content={"detail": "Error interno del servicio."})
        if tracked and response.status_code != 307:
            append_event(log_path, response.status_code,
                         getattr(request.state, "model_version", None),
                         getattr(request.state, "prediction", None))
        return response

    @application.get("/health")
    def health():
        return {"estado": "ok"}

    @application.post("/predict")
    def predict(request: Request, record: object = Body(...)):
        values = validate_record(record)
        try:
            model = load_model(models_dir / model_version)
        except ModelLoadError as exc:
            raise HTTPException(503, f"Version '{model_version}' no disponible: {exc}") from exc
        request.state.model_version = model_version
        try:
            result = predict_dataframe(model, pd.DataFrame([values], columns=FEATURE_COLUMNS))
            prediction = float(result[PREDICTION_COLUMN].iloc[0])
            if not math.isfinite(prediction):
                raise ValueError("El modelo produjo un valor no finito.")
        except DataValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        request.state.prediction = prediction
        return {"prediccion": prediction, "version": model_version}

    @application.post("/train", status_code=201)
    def train(configuration: dict | None = Body(default=None)):
        if configuration:
            raise HTTPException(422, "Envie un cuerpo vacio o {}: se usa la configuracion de fase 2.")
        version = datetime.now(timezone.utc).strftime("v-%Y%m%dT%H%M%S-") + uuid4().hex[:12]
        options = {"seed": 42, "log_target": True, **(training_options or {})}
        try:
            result = run_training(data_path, models_dir, version, **options)
        except (DataValidationError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"version": version, "metricas": result["metrics"]}

    return application


app = create_app()
