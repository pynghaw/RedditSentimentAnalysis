from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- 1. Load the pre-trained model AND tokenizer ---
# Define the max length of sequences from your training phase
MAX_SEQUENCE_LENGTH = 100 # IMPORTANT: Use the same value you used for training!

def load_files():
    # Load the Keras model
    model = tf.keras.models.load_model("/app/artifacts/lstm_sentiment_model.h5")
    
    # Load the tokenizer
    with open("/app/artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

# Broadcast the model and tokenizer to all worker nodes
model, tokenizer = load_files()

# --- 2. Define the Prediction UDF for LSTM ---
# A User Defined Function (UDF) to apply the LSTM model
def predict_sentiment_lstm(text):
    if text is None:
        return "neutral"
        
    try:
        # 1. Tokenize the text using the loaded tokenizer
        sequence = tokenizer.texts_to_sequences([text])
        
        # 2. Pad the sequence to the same length used during training
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        
        # 3. Predict the sentiment
        prediction = model.predict(padded_sequence)
        
        # 4. Interpret the result (assuming 3 output neurons for pos, neg, neu)
        #    and return the label with the highest probability.
        labels = ['negative', 'neutral', 'positive'] # Make sure this order matches your model's training
        return labels[np.argmax(prediction)]
        
    except Exception as e:
        return "error"

# Register the function as a UDF
sentiment_udf = udf(predict_sentiment_lstm, StringType())

# --- 3. Initialize Spark Session ---
print("Initializing Spark Session for LSTM model...")
spark = SparkSession.builder \
    .appName("RedditSentimentAnalysisLSTM") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.12.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session Initialized.")

# --- 4. Define Schema and Read from Kafka ---
schema = StructType() \
    .add("post_id", StringType()) \
    .add("post_title", StringType()) \
    .add("comment_id", StringType()) \
    .add("comment_body", StringType()) \
    .add("comment_score", IntegerType()) \
    .add("created_utc", FloatType())

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "reddit-stream") \
    .option("startingOffsets", "latest") \
    .load()

# --- 5. Process Data and Predict Sentiment ---
df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

df_with_sentiment = df_parsed.withColumn("sentiment", sentiment_udf(col("comment_body")))

# --- 6. Write to Elasticsearch ---
print("Starting to write stream to Elasticsearch...")
query = df_with_sentiment.writeStream \
    .outputMode("append") \
    .format("org.elasticsearch.spark.sql") \
    .option("es.resource", "reddit_sentiment_lstm/comment") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("checkpointLocation", "/tmp/check_point_reddit_lstm") \
    .start()

query.awaitTermination()