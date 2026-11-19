import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def read_record(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        record = next(csv.DictReader(stream), None)
    if record is None:
        raise ValueError("El CSV no contiene registros.")
    record.pop("Price", None)
    for name, value in record.items():
        if value is None or value.strip() == "":
            record[name] = None
        elif name in ("Year", "Mileage", "Cylinders"):
            try:
                record[name] = float(value)
            except ValueError:
                if name != "Cylinders":
                    raise ValueError(f"El campo '{name}' debe ser numerico.")
    return record


def main():
    parser = argparse.ArgumentParser(description="Envia un vehiculo del CSV a la API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--input", default=ROOT / "data/ejemplo_prediccion.csv")
    parser.add_argument("--invalid", action="store_true", help="Omite Mileage para probar la validacion.")
    args = parser.parse_args()
    try:
        record = read_record(args.input)
        if args.invalid:
            record.pop("Mileage", None)
        request = Request(args.url.rstrip("/") + "/predict",
                          data=json.dumps(record, allow_nan=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            print(f"HTTP {response.status}")
            print(response.read().decode("utf-8"))
    except HTTPError as exc:
        print(f"HTTP {exc.code}")
        print(exc.read().decode("utf-8"))
        return 1
    except (OSError, ValueError, URLError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
