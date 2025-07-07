# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your producer script into the container
COPY ./kafka_spark_pipeline/kafka_producer.py .

# Command to run when the container starts
CMD ["python", "kafka_producer.py"]