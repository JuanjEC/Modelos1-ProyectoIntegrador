# Imagen base con version explicita de Python
FROM python:3.14.2

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
