# Fase 1: Modelo predictivo. Precio de vehiculos usados en UAE

Modelos y Simulacion de Sistemas I, Universidad de Antioquia, 2026-2. Docente: Andres Parra.

**Integrantes:** Andres Felipe Borja Usuga, Juan Jeronimo Echeverry Chica, Cristian Mateo Sepulveda Ochoa.

## Descripcion del problema

Regresion: predecir el precio (`Price`, en AED) de un vehiculo usado a partir de sus caracteristicas.

## Fuente del conjunto de datos

Kaggle, "UAE Used Cars Analysis": https://www.kaggle.com/datasets/mohamedsaad254/uae-used-cars-analysis-full-project-v1-0 (archivo `uae_used_cars_10k.csv`, 10.000 filas, 11 variables predictoras). Contiene 105 nulos en `Cylinders` (1,05 %) y ademas 50 valores `Unknown` en esa misma columna.

## Objetivo del modelo

Estimar el precio de un vehiculo usado y compararlo contra un modelo base.

## Algoritmo

Pipeline de scikit-learn: imputacion (mediana / moda), `StandardScaler`, `OneHotEncoder`, TF-IDF sobre `Description` (300 terminos) y `RandomForestRegressor` (120 arboles, profundidad 20). Modelo base: `DummyRegressor` con la mediana. Seleccion por validacion cruzada de 3 particiones usando solo el entrenamiento (Dummy, Ridge, Random Forest).

## Metrica

MAE (principal), RMSE y R2, sobre un conjunto de prueba del 20 % (semilla 42).

## Resultados

| Metrica | Random Forest |
|---|---|
| MAE | 118341.93 |
| RMSE | 388796.19 |
| R2 | 0.42 |

Fuga de informacion: la separacion train/test se hace antes de ajustar cualquier transformacion; todo el preprocesamiento esta dentro de un `Pipeline` ajustado solo con entrenamiento; `Price` se usa solo como objetivo.

## Contenido de la carpeta

- `notebook.ipynb`: analisis, preparacion, modelos, evaluacion y guardado.
- `modelo.joblib`: pipeline completo entrenado.
- `README.md`: este archivo.

# Fase 2: Scripts y Docker. Prediccion de precios de vehiculos usados (UAE)

**Entrega 2, commit o etiqueta:** `entrega-2` (completar con el hash del commit tras crear la etiqueta).

## Problema y relacion con la Fase 1

Regresion: predecir `Price` (AED) de un vehiculo usado a partir de 11 variables (`Make`, `Model`, `Year`, `Mileage`, `Body Type`, `Cylinders`, `Transmission`, `Fuel Type`, `Color`, `Location`, `Description`).
Se conservan el dataset y el enfoque de la Fase 1 (Pipeline con imputacion, escalado, One-Hot, TF-IDF y Random Forest; baseline `DummyRegressor` con la mediana). El notebook sigue en `../fase-1/`.

## Cambios respecto a la Fase 1

- El notebook se convirtio en scripts: `train.py` y `predict.py`, con la logica compartida en `src/`.
- La limpieza determinista (`Cylinders` no numerico a nulo, `Description` nula a texto vacio) ahora **vive dentro del Pipeline**, asi el artefacto guardado es autosuficiente.
- Correccion de calidad de datos: `Location` traia espacios al inicio (`" Dubai"` y `"Dubai"` se contaban como categorias distintas, 16 en total en vez de 8). Ahora se hace `strip` en todas las categoricas. Por eso el MAE de v1 (118404) difiere levemente del de la Fase 1 (118342).
- Se agrega la opcion `--log-target` (entrena sobre log(1+Price)), usada en v2, porque `Price` es muy asimetrico y asi la prediccion nunca es negativa.
- No hay identificadores de grupo y no existen filas repetidas en las variables predictoras, por eso la separacion es aleatoria 80/20 (justificado en `src/training.py`).

## Requisitos y entorno

- **Python 3.14.2** (el mismo en local y en Docker). Dependencias con versiones fijas en `requirements.txt`.

Bash (Linux/macOS):
```bash
Python 3.14.2 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
PowerShell (Windows):
```powershell
py -3.14.2 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos

- Fuente: Kaggle, "UAE Used Cars Analysis" https://www.kaggle.com/datasets/mohamedsaad254/uae-used-cars-analysis-full-project-v1-0 (archivo `uae_used_cars_10k.csv`, 10.000 filas).
- El CSV (1,9 MB) esta incluido en `data/uae_used_cars_10k.csv`. Si no estuviera, descargarlo de Kaggle y dejarlo en esa ruta.
- Esquema del CSV de prediccion (`data/ejemplo_prediccion.csv`): las 11 columnas predictoras, **sin** `Price`. El orden de columnas no importa.

| Columna | Tipo | Nulos |
|---|---|---|
| Year, Mileage | numerico | se imputan con la mediana de entrenamiento |
| Cylinders | numerico (acepta "Unknown", que pasa a nulo) | mediana |
| Make, Model, Body Type, Transmission, Fuel Type, Color, Location | texto | moda de entrenamiento |
| Description | texto libre | texto vacio |

Reglas de prediccion: columna requerida ausente o `Year`/`Mileage` no numericos dan error claro y codigo de salida distinto de cero (no se genera el CSV). Columnas extra se ignoran (y se conservan en la salida). Categorias no vistas en entrenamiento se ignoran (One-Hot en ceros). Se predice una fila por fila de entrada, en el mismo orden.

## Uso

Todos los comandos se ejecutan dentro de `fase-2/`, en **una sola linea**.

```bash
python train.py --help
python predict.py --help

python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v1 --seed 42
python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v2 --seed 42 --log-target

python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --model models/v2/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```
Repetir un `--version` existente se rechaza con un error (no se sobrescribe).
Salida: `outputs/predicciones.csv` = columnas de entrada + `prediccion` (precio en AED).

## Versiones del modelo

```
models/
  v1/  modelo.joblib  metrics.json  metadata.json   # Random Forest (como Fase 1)
  v2/  modelo.joblib  metrics.json  metadata.json   # Random Forest con --log-target
```

## Resultados (conjunto de prueba, 2.000 filas, semilla 42)

Metrica principal: MAE (AED), con RMSE y R2. Baseline: predecir siempre la mediana del precio de entrenamiento.

| Modelo | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline (mediana) | 192459.92 | 532043.08 | -0.0797 |
| v1 Random Forest | 118404.02 | 388746.29 | 0.4236 |
| v2 Random Forest + log(1+Price) | 109280.51 | 385337.72 | 0.4336 |

Ambas versiones mejoran al baseline. El R2 moderado indica que el precio tiene mucha variacion que las variables no explican.

**Limitaciones:** `Price` muy asimetrico con vehiculos de lujo muy caros (el RMSE es alto por eso); `Description` repite informacion de otras columnas; una sola particion train/test (sin validacion cruzada en los scripts); el Random Forest tiene componentes aleatorios, fijados con la semilla, pero pueden aparecer diferencias numericas minimas entre sistemas.

## Pruebas

```bash
python -m pytest
```
8 pruebas con datos sinteticos y `tmp_path`: (1) entrenamiento y archivos de la version, (2) carga y prediccion igual al modelo en memoria, (3) valores faltantes, (4) columna ausente y tipo invalido (incluye la CLI con codigo de salida distinto de cero), (5) versionamiento (segunda version conserva la primera, id repetido no sobrescribe), y (6) categorias nuevas, columnas extra y orden de columnas.

## Docker

Probado con terminal: _completar (Bash o PowerShell)_.

Bash:
```bash
docker build -t modelos-fase2:1.0 .
mkdir -p models outputs
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/uae_used_cars_10k.csv --target Price --output-dir models --version v-docker --seed 42
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
docker run --rm modelos-fase2:1.0 python -m pytest -q
```
PowerShell: igual, cambiando `$PWD` por `${PWD}` y `mkdir -p` por `mkdir models, outputs -Force`.

El entrenamiento NO ocurre en `docker build`; `models/` y `outputs/` persisten en el equipo anfitrion por los montajes.

## Evidencia de ejecucion

_Completar con la salida real de pytest, de train.py, de predict.py y de Docker (capturas o texto)._

## Fuentes

Documentacion de scikit-learn (Pipeline, ColumnTransformer, TransformedTargetRegressor), dataset de Kaggle citado arriba. Codigo adaptado del notebook de la Fase 1 del equipo.
