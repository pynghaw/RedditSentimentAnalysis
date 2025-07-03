from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaSparkSentiment") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define JSON schema
schema = StructType() \
    .add("text", StringType()) \
    .add("source", StringType())

# Read stream from Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "reddit-sentiment") \
    .option("startingOffsets", "earliest") \
    .load()

# Extract JSON
df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# Print to console
query = df_parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
