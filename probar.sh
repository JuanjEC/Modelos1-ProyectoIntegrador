#!/usr/bin/env bash
# Verificacion de la Fase 2. Ejecutar desde la raiz del repositorio:
#   bash probar.sh 2>&1 | tee resultado.txt

echo "== 1. Python y pytest"
python --version
python -m pytest -q

echo
echo "== 2. CSV de ejemplo para predecir (sin Price)"
if [ ! -f data/ejemplo_prediccion.csv ]; then
  python -c "import pandas as pd; df=pd.read_csv('data/uae_used_cars_10k.csv'); df.drop(columns=['Price']).sample(5,random_state=1).to_csv('data/ejemplo_prediccion.csv',index=False)"
  echo "creado data/ejemplo_prediccion.csv"
else
  echo "ya existe data/ejemplo_prediccion.csv"
fi

echo
echo "== 3. Prediccion local con v1 (ruta .joblib) y v2 (carpeta)"
mkdir -p outputs
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/pred_v1.csv
echo "codigo de salida: $?"
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/pred_v2.csv
echo "codigo de salida: $?"
head -n 3 outputs/pred_v1.csv outputs/pred_v2.csv

echo
echo "== 4. Version repetida (debe rechazarse, codigo 1)"
python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v1 --seed 42
echo "codigo de salida: $?"

echo
echo "== 5. Modelo inexistente (debe fallar, codigo 1, sin crear el CSV)"
python predict.py --model models/no-existe/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/no_debe_existir.csv
echo "codigo de salida: $?"
ls outputs/no_debe_existir.csv 2>&1

echo
echo "== 6. Metadatos de v1 y v2 (python_version y git_commit)"
grep -H -E '"python_version"|"git_commit"' models/v1/metadata.json models/v2/metadata.json

echo
echo "== 7. Docker: build"
docker build -t modelos-fase2:1.0 .
echo "codigo de salida: $?"

echo
echo "== 8. Docker: entrenamiento (version v-docker)"
mkdir -p models outputs
docker run --rm -e GIT_COMMIT="$(git rev-parse HEAD)" -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v-docker --seed 42
echo "codigo de salida: $?"

echo
echo "== 9. Docker: prediccion con v-docker"
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/pred_docker.csv
echo "codigo de salida: $?"

echo
echo "== 10. Persistencia fuera del contenedor"
ls -l models/v-docker outputs/pred_docker.csv

echo
echo "== 11. pytest dentro de Docker"
docker run --rm modelos-fase2:1.0 python -m pytest -q
echo "codigo de salida: $?"

echo
echo "== Fin. Para borrar lo de prueba: sudo rm -rf models/v-docker"
