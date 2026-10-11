# Imagen base con version explicita de Python (la misma usada en local: 3.12)
FROM python:3.12-slim

WORKDIR /app

# Dependencias primero (aprovecha la cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo necesario. NO se entrena nada durante el build.
COPY src/ ./src/
COPY tests/ ./tests/
COPY train.py predict.py ./

# Sin ENTRYPOINT: el comando (train.py, predict.py o pytest) se indica en docker run.
CMD ["python", "train.py", "--help"]
