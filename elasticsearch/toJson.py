import pandas as pd

# Load the CSV file
csv_file = 'balanced_dataset.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Convert the DataFrame to JSON
json_file = 'reddit_data.json'  # Output JSON file name
df.to_json(json_file, orient='records', lines=True)

print(f"CSV file has been converted to JSON and saved as {json_file}")
