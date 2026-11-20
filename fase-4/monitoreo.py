import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring import summarize


def main():
    parser = argparse.ArgumentParser(description="Metricas operativas del registro de predicciones.")
    parser.add_argument("--log", default=ROOT / "outputs/solicitudes.jsonl")
    args = parser.parse_args()
    try:
        result = summarize(args.log)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Total de solicitudes: {result['total_solicitudes']}")
    print(f"Solicitudes fallidas: {result['fallidas']}")
    percentage = result["porcentaje_fallos"]
    text = "sin solicitudes" if percentage is None else f"{percentage:.1f} %".replace(".", ",")
    print(f"Porcentaje de fallos: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
