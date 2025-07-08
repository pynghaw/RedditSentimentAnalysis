# 📡 Real-Time Reddit Comment Sentiment Analysis System (with LSTM)

This project sets up a real-time sentiment analysis pipeline for Reddit comments using a **Keras LSTM model**. The backend uses **Apache Kafka**, **Apache Spark**, **Elasticsearch**, and **Kibana**, all orchestrated with **Docker Compose**.

---

## 🛠️ Setup Instructions

### 🔧 Prerequisites

-   A Reddit account and API credentials (Client ID and Secret).
-   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/).
-   Python 3.8+ and `pip`.
-   Your trained model files (`lstm_sentiment_model.h5`, `tokenizer.pkl`) in a `models/` folder.

### ✅ Initial Setup

1.  **Install Python Libraries for the Producer:**
    ```bash
    pip install praw kafka-python python-dotenv
    ```

2.  **Create `.env` file:**
    Create a file named `.env` in your project folder and add your Reddit API credentials:
    ```
    REDDIT_CLIENT_ID="YOUR_CLIENT_ID"
    REDDIT_CLIENT_SECRET="YOUR_CLIENT_SECRET"
    REDDIT_USER_AGENT="MyRedditStreamer/0.1 by YourUsername"
    ```

---

## 🚀 How to Run the System

### 1️⃣ Start Docker Services

In your terminal, launch all the services. Make sure Docker Desktop is running.

```bash
docker-compose up -d