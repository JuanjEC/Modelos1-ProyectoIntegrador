import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_LOCK = Lock()


def append_event(log_path, status, version=None, prediction=None):
    event = {
        "fecha_hora": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado_http": int(status),
        "version": version,
        "prediccion": prediction,
    }
    path = Path(log_path)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(log_path):
    path = Path(log_path)
    total = failed = 0
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    status = event["estado_http"]
                    if type(status) is not int or not 100 <= status <= 599:
                        raise ValueError("estado_http invalido")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Registro invalido en la linea {line_number}: {exc}") from exc
                total += 1
                failed += 400 <= status <= 599
    return {
        "total_solicitudes": total,
        "fallidas": failed,
        "porcentaje_fallos": 100.0 * failed / total if total else None,
    }
