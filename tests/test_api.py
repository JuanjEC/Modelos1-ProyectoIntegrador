import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_aplicacion import FAST, make_df
from src.training import run_training

spec = importlib.util.spec_from_file_location("project_api", ROOT / "fase-3/api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


@pytest.fixture()
def setup_api(tmp_path):
    data = tmp_path / "datos.csv"
    make_df().to_csv(data, index=False)
    models = tmp_path / "models"
    run_training(data, models, "v-test", **FAST)
    log = tmp_path / "solicitudes.jsonl"
    app = api.create_app(models_dir=models, model_version="v-test", data_path=data,
                         log_path=log, training_options=FAST)
    with TestClient(app) as client:
        yield client, log, models


@pytest.fixture()
def record():
    return make_df(1, with_target=False).iloc[0].to_dict()


def events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_health(setup_api):
    client, log, _ = setup_api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"estado": "ok"}
    assert not log.exists()


def test_predict_returns_prediction_and_version(setup_api, record):
    client, log, _ = setup_api
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["prediccion"], float)
    assert result["version"] == "v-test"
    assert events(log)[0]["prediccion"] == result["prediccion"]
    assert events(log)[0]["version"] == "v-test"


def test_required_field_is_rejected_and_logged(setup_api, record):
    client, log, _ = setup_api
    del record["Mileage"]
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert "Mileage" in response.text
    assert len(events(log)) == 1
    assert events(log)[0]["estado_http"] == 422
    assert events(log)[0]["version"] is None


@pytest.mark.parametrize("name,value", [("Year", "2020"), ("Mileage", True),
                                        ("Make", 10), ("Cylinders", []),
                                        ("Description", {})])
def test_incompatible_types(setup_api, record, name, value):
    client, log, _ = setup_api
    record[name] = value
    response = client.post("/predict", json=record)
    assert response.status_code == 422
    assert name in response.text
    assert len(events(log)) == 1


def test_nulls_use_saved_preprocessing(setup_api, record):
    client, _, _ = setup_api
    record = dict.fromkeys(record)
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    assert isinstance(response.json()["prediccion"], float)


def test_unknown_cylinders_new_category_and_extra_target(setup_api, record):
    client, _, _ = setup_api
    record.update(Cylinders="Unknown", Make="nueva-marca", Price=1)
    assert client.post("/predict", json=record).status_code == 200


@pytest.mark.parametrize("body", [None, [], "texto"])
def test_invalid_json_shape_is_logged(setup_api, body):
    client, log, _ = setup_api
    assert client.post("/predict", json=body).status_code == 422
    assert len(events(log)) == 1


def test_malformed_json_is_logged(setup_api):
    client, log, _ = setup_api
    response = client.post("/predict", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert len(events(log)) == 1


def test_missing_model_returns_503(tmp_path, record):
    log = tmp_path / "log.jsonl"
    client = TestClient(api.create_app(models_dir=tmp_path, model_version="ausente", log_path=log))
    response = client.post("/predict", json=record)
    assert response.status_code == 503
    assert "ausente" in response.text
    assert events(log)[0]["estado_http"] == 503
    assert events(log)[0]["version"] is None


def test_corrupt_model_returns_503(tmp_path, record):
    model = tmp_path / "corrupto/modelo.joblib"
    model.parent.mkdir()
    model.write_bytes(b"no es un modelo")
    log = tmp_path / "log.jsonl"
    client = TestClient(api.create_app(models_dir=tmp_path, model_version="corrupto", log_path=log))
    assert client.post("/predict", json=record).status_code == 503
    assert len(events(log)) == 1


def test_train_preserves_versions_and_does_not_switch_active(setup_api, record):
    client, log, models = setup_api
    previous = (models / "v-test/modelo.joblib").read_bytes()
    first = client.post("/train")
    second = client.post("/train", json={})
    assert first.status_code == second.status_code == 201
    version = first.json()["version"]
    assert second.json()["version"] != version
    for filename in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (models / version / filename).is_file()
    assert "MAE" in first.json()["metricas"]["modelo"]
    assert (models / "v-test/modelo.joblib").read_bytes() == previous
    assert not log.exists()
    assert client.post("/predict", json=record).json()["version"] == "v-test"


def test_two_valid_one_invalid_append_across_restarts(setup_api, record):
    client, log, models = setup_api
    assert client.post("/predict", json=record).status_code == 200
    assert client.post("/predict", json=record).status_code == 200
    del record["Year"]
    assert client.post("/predict", json=record).status_code == 422
    assert [event["estado_http"] for event in events(log)] == [200, 200, 422]
    other = TestClient(api.create_app(models_dir=models, model_version="v-test", log_path=log))
    assert other.post("/predict", json=record).status_code == 422
    assert len(events(log)) == 4


def test_unhandled_prediction_error_is_logged(setup_api, record, monkeypatch):
    client, log, _ = setup_api

    def fail(*args, **kwargs):
        raise RuntimeError("fallo de prueba")

    monkeypatch.setattr(api, "predict_dataframe", fail)
    response = client.post("/predict", json=record)
    assert response.status_code == 500
    assert len(events(log)) == 1
    assert events(log)[0]["estado_http"] == 500
    assert events(log)[0]["prediccion"] is None
