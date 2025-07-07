import praw
from kafka import KafkaProducer
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("Initializing Reddit API...")
# Reddit API credentials from environment variables
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
print("Reddit API Initialized.")

print("Initializing Kafka Producer...")
# Kafka configuration
producer = KafkaProducer(
    bootstrap_servers="kafka:29092", # Connect to Kafka inside Docker
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)
print("Kafka Producer Initialized.")

# Choose subreddit - using a high-traffic one for demonstration
subreddit = reddit.subreddit("malaysia")
KAFKA_TOPIC = "reddit-stream"

print(f"Starting to stream comments from r/{subreddit.display_name} to topic '{KAFKA_TOPIC}'...")

while True:
    try:
        # Stream comments in real-time
        for comment in subreddit.stream.comments(skip_existing=True):
            # Prepare the message
            message = {
                "post_id": comment.submission.id,
                "post_title": comment.submission.title,
                "comment_id": comment.id,
                "comment_body": comment.body,
                "comment_score": comment.score,
                "created_utc": comment.created_utc
            }

            # Send to Kafka
            producer.send(KAFKA_TOPIC, message)
            print(f"Sent comment ID {comment.id}")

            # Optional: small delay to avoid getting rate-limited
            time.sleep(0.1)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Retrying in 60 seconds...")
        time.sleep(60)