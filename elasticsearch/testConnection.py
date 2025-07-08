from elasticsearch import Elasticsearch

# Connect to Elasticsearch (make sure it is running on localhost:9200)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Check if the connection is successful
if es.ping():
    print("Successfully connected to Elasticsearch!")
else:
    print("Failed to connect to Elasticsearch!")
