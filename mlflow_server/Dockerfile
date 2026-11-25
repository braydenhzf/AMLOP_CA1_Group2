
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem', '/app/global-bundle.pem')"

COPY start_mlflow.py .

ENV AWS_DEFAULT_REGION=us-east-1

EXPOSE 5000

CMD ["python", "start_mlflow.py"]
