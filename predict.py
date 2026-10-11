"""Genera predicciones con una version guardada del modelo.

Ejemplo (una sola linea):
  python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""

import argparse
import sys

from src.errors import AppError
from src.prediction import run_prediction


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prediccion de precios (AED) con una version guardada.")
    p.add_argument("--model", required=True, help="Ruta del modelo (.joblib) o de la carpeta de la version.")
    p.add_argument("--input", required=True, help="CSV con las variables de entrada (sin Price).")
    p.add_argument("--output", required=True, help="CSV de salida con la columna 'prediccion'.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        res = run_prediction(args.model, args.input, args.output)
    except AppError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Predicciones generadas: {res['n_rows']} filas -> {res['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
