from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WriteToElasticsearch") \
    .getOrCreate()

# Load your static dataset (e.g., CSV or JSON)
# Replace this with the actual path to your dataset
df = spark.read.csv("C:/Users/Chen Pyng Haw/Projects/SentimentAnalysis/elasticsearch/reddit_data.csv", header=True, inferSchema=True)

# Specify Elasticsearch connection details
ELASTICSEARCH_NODE = "localhost"  # or Docker service name
ELASTICSEARCH_PORT = "9200"
ELASTICSEARCH_INDEX = "reddit-comments-lstm"  # Elasticsearch index name

# Write static dataset to Elasticsearch
df.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", ELASTICSEARCH_NODE) \
    .option("es.port", ELASTICSEARCH_PORT) \
    .option("es.resource", ELASTICSEARCH_INDEX) \
    .save()

print(f"Data successfully written to Elasticsearch index: {ELASTICSEARCH_INDEX}")
