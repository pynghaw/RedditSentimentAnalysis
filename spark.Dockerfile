# Start from the official Bitnami Spark image
FROM bitnami/spark:latest

# Switch to the root user to have permission to install packages
USER root

# Copy your requirements file into a temporary location
COPY requirements.txt /tmp/requirements.txt

# Install the python dependencies from the requirements file
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch back to the default non-root user for better security
USER 1001