FROM python:3.9-slim

WORKDIR /app

COPY kafka_spark_pipeline/ ./kafka_spark_pipeline/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "kafka_spark_pipeline/kafka_producer.py"]
