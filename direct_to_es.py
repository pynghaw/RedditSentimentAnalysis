import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# --- Configuration ---
ELASTICSEARCH_HOST = "localhost" # Connect to the port exposed on your machine
ELASTICSEARCH_PORT = 9200
ELASTICSEARCH_INDEX = "reddit-comments" # New index for this method
INPUT_JSON_PATH = "reddit_data.json" # Path to your local JSON file

def generate_actions(file_path, index_name):
    """
    Reads a JSON file line by line and yields Elasticsearch bulk actions.
    This assumes your JSON file has one complete JSON object per line.
    """
    print(f"📖 Reading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Load the JSON object from the line
                doc = json.loads(line)
                # Yield a document formatted for the bulk API
                yield {
                    "_index": index_name,
                    "_source": doc,
                }
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not decode line, skipping: {line.strip()}")
                continue

def main():
    """
    Main function to connect to ES and upload data.
    """
    print("🚀 Starting direct upload to Elasticsearch")

    # 1. Connect to Elasticsearch
    try:
        es_client = Elasticsearch(
            [f"http://{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}"]
        )
        if es_client.ping():
            print("✅ Connected to Elasticsearch successfully.")
        else:
            print("❌ Could not connect to Elasticsearch.")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    # 2. Use the bulk helper to upload the data
    try:
        print(f"✍️ Writing data to Elasticsearch index: {ELASTICSEARCH_INDEX}")
        # The bulk helper is highly efficient for uploading many documents.
        successes, failures = bulk(
            es_client,
            generate_actions(INPUT_JSON_PATH, ELASTICSEARCH_INDEX)
        )
        print(f"✅ Upload complete! Successes: {successes}, Failures: {failures}")

    except Exception as e:
        print(f"❌ An error occurred during bulk upload: {e}")

if __name__ == "__main__":
    main()