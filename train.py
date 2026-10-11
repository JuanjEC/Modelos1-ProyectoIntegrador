"""Entrena una version del modelo y la guarda en <output-dir>/<version>/.

Ejemplo (una sola linea):
  python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v1 --seed 42
"""

import argparse
import sys

from src.config import DEFAULT_TARGET
from src.errors import AppError
from src.modeling import ALGORITHMS
from src.training import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entrenamiento del modelo de precios de vehiculos usados (UAE).")
    p.add_argument("--data", required=True, help="Ruta del CSV de datos etiquetados.")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Nombre de la variable objetivo (por defecto: Price).")
    p.add_argument("--output-dir", required=True, help="Directorio donde se guardan las versiones (ej. models).")
    p.add_argument("--version", required=True, help="Identificador de la version (ej. v1). No se sobrescribe.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (por defecto: 42).")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraccion para prueba (por defecto: 0.2).")
    p.add_argument("--algorithm", choices=ALGORITHMS, default="random_forest", help="Algoritmo.")
    p.add_argument("--log-target", action="store_true", help="Entrena sobre log(1+precio) y devuelve AED.")
    p.add_argument("--n-estimators", type=int, default=120, help="Arboles del Random Forest.")
    p.add_argument("--max-depth", type=int, default=20, help="Profundidad maxima del Random Forest.")
    p.add_argument("--min-samples-leaf", type=int, default=2, help="Minimo de muestras por hoja (RF).")
    p.add_argument("--alpha", type=float, default=10.0, help="Regularizacion de Ridge.")
    p.add_argument("--max-tfidf-features", type=int, default=300, help="Maximo de terminos TF-IDF.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        res = run_training(
            data_path=args.data, output_dir=args.output_dir, version=args.version,
            target=args.target, seed=args.seed, test_size=args.test_size,
            algorithm=args.algorithm, log_target=args.log_target,
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf, alpha=args.alpha,
            max_tfidf_features=args.max_tfidf_features,
        )
    except (AppError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m, b = res["metrics"]["modelo"], res["metrics"]["baseline_mediana"]
    print(f"Version '{args.version}' entrenada ({args.algorithm}, log_target={args.log_target})")
    print(f"  Modelo   -> MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")
    print(f"  Baseline -> MAE={b['MAE']:.2f}  RMSE={b['RMSE']:.2f}  R2={b['R2']:.4f}")
    for name, path in res["paths"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
