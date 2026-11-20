import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring import append_event, summarize


def test_empty_or_missing_log(tmp_path):
    path = tmp_path / "registro.jsonl"
    expected = {"total_solicitudes": 0, "fallidas": 0, "porcentaje_fallos": None}
    assert summarize(path) == expected
    path.touch()
    assert summarize(path) == expected


def test_two_successes_and_one_failure(tmp_path):
    path = tmp_path / "logs/registro.jsonl"
    append_event(path, 200, "v2", 25000.0)
    append_event(path, 200, "v2", 30000.0)
    append_event(path, 422)
    result = summarize(path)
    assert result["total_solicitudes"] == 3
    assert result["fallidas"] == 1
    assert result["porcentaje_fallos"] == pytest.approx(100 / 3)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert set(rows[0]) == {"fecha_hora", "estado_http", "version", "prediccion"}
    assert rows[-1]["version"] is rows[-1]["prediccion"] is None
    assert "+00:00" in rows[0]["fecha_hora"]


def test_counts_4xx_and_5xx_only(tmp_path):
    path = tmp_path / "registro.jsonl"
    for status in (200, 302, 400, 422, 500, 503):
        append_event(path, status)
    assert summarize(path)["fallidas"] == 4
    assert summarize(path)["total_solicitudes"] == 6


@pytest.mark.parametrize("line", ['{', '{}', '{"estado_http": "200"}', 'null',
                                 '{"estado_http": 999}', '{"estado_http": true}'])
def test_corrupt_log_fails_clearly(tmp_path, line):
    path = tmp_path / "registro.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="linea 1"):
        summarize(path)


def test_monitoring_cli(tmp_path):
    path = tmp_path / "registro.jsonl"
    for status in (200, 200, 422):
        append_event(path, status)
    result = subprocess.run([sys.executable, str(ROOT / "fase-4/monitoreo.py"), "--log", str(path)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0
    assert "Total de solicitudes: 3" in result.stdout
    assert "33,3 %" in result.stdout
