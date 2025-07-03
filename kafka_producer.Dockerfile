FROM python:3.9-slim

WORKDIR /app

COPY kafka_spark_pipeline/kafka_producer.py .

RUN pip install kafka-python

CMD ["python", "kafka_producer.py"]
