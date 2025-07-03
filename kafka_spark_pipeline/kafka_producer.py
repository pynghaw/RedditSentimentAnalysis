from kafka import KafkaProducer
import json
import time

# Kafka configuration
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Example messages (replace with real Reddit data later)
messages = [
    {"text": "This is amazing!", "source": "reddit"},
    {"text": "Terrible experience, very bad.", "source": "reddit"},
    {"text": "Just okay, nothing special.", "source": "reddit"},
]

# Send messages
for msg in messages:
    producer.send("reddit-sentiment", msg)
    print(f"Sent: {msg}")
    time.sleep(2)

producer.flush()
