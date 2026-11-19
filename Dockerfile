FROM python:3.14.2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY fase-3/ ./fase-3/
COPY fase-4/ ./fase-4/
COPY train.py predict.py ./

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "--app-dir", "fase-3", "api:app", "--host", "0.0.0.0", "--port", "8000"]
