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