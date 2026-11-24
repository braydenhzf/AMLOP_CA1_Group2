
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/mlruns /app/mlartifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////app/mlruns/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/app/mlartifacts"]
