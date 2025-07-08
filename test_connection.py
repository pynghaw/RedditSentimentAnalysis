import requests

url = "http://localhost:9200"

print(f"Attempting to connect to {url}...")

try:
    response = requests.get(url, timeout=5) # 5 second timeout
    response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)

    print("✅ Connection successful!")
    print("Response JSON:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"❌ Connection failed: {e}")