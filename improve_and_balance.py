import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.utils import resample
from tqdm import tqdm
tqdm.pandas()


def bert_sentiment_to_label(result):
    try:
        label = result.get('label', 'unknown') if isinstance(result, dict) else 'unknown'

        if label in ['1 star', '2 stars']:
            return 'negative'
        elif label == '3 stars':
            return 'neutral'
        elif label in ['4 stars', '5 stars']:
            return 'positive'
        else:
            # Fallback for short labels like 'p', 'n'
            label = label.lower()
            if label in ['p', 'pos', 'positive']:
                return 'positive'
            elif label in ['n', 'neg', 'negative']:
                return 'negative'
            elif label in ['neu', 'neutral']:
                return 'neutral'
            else:
                return 'neutral'
    except Exception as e:
        return 'neutral'


# Try loading intermediate results to save time
input_file = r"C:\Users\ccmar\reddit_movies_sentiment_ready.csv"  # Update this path
intermediate_file = "intermediate_after_sentiment.csv"
output_file = "balanced_dataset.csv"

print("🔄 Loading dataset...")

try:
    df = pd.read_csv(intermediate_file)
    print("✅ Loaded pre-labeled data")
except FileNotFoundError:
    print("🔄 Running sentiment analysis...")

    # Load raw data
    df = pd.read_csv(input_file)

    # Drop empty comments
    df = df[df['cleaned_comment'].notna() & (df['cleaned_comment'] != '')]

    # Clean text function
    def clean_text(text):
        text = str(text)
        text = re.sub(r'http\S+', '', text)                 # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()             # Normalize spaces
        return text

    df['cleaned_comment'] = df['cleaned_comment'].apply(clean_text)

    # Rename sentiment column if exists
    if 'sentiment' in df.columns:
        df = df.drop(columns=['sentiment'], errors='ignore')

    # Load HuggingFace sentiment analysis pipeline
    print("🧠 Loading transformer sentiment model (BERT-based)...")

    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to {'cuda' if device == 0 else 'cpu'}")

    nlp = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=32
    )

    # Truncate long texts to max 512 tokens
    def truncate_text(text, max_length=512):
        return tokenizer.decode(tokenizer.encode(text, max_length=max_length, truncation=True), skip_special_tokens=True)

    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: truncate_text(x))

    # Batch prediction
    texts = df['cleaned_comment'].tolist()
    print("🏷️ Improving sentiment labels with HuggingFace Transformer...")

    results = nlp(texts)
    labels = [bert_sentiment_to_label(res) for res in results]
    df['sentiment'] = labels

    # Save intermediate result
    df.to_csv(intermediate_file, index=False)
    print(f"💾 Intermediate labeled data saved to '{intermediate_file}'")

# Count original class distribution
class_counts = df['sentiment'].value_counts()
print("\n📊 Original Class Distribution:")
print(class_counts)

# Split by class
df_neg = df[df['sentiment'] == 'negative']
df_neu = df[df['sentiment'] == 'neutral']
df_pos = df[df['sentiment'] == 'positive']

available_classes = []
if len(df_neg) > 0:
    available_classes.append('negative')
if len(df_neu) > 0:
    available_classes.append('neutral')
if len(df_pos) > 0:
    available_classes.append('positive')

if not available_classes:
    print("⚠️ No valid classes found after filtering.")
    balanced_df = df.assign(sentiment='neutral')  # fallback
else:
    class_sizes = [len(df[df['sentiment'] == c]) for c in available_classes if len(df[df['sentiment'] == c]) > 0]
    min_size = min(class_sizes) if class_sizes else 1

    print(f"\n✂️ Balancing dataset to {min_size} samples per class: {available_classes}")

    balanced_dfs = []
    if 'negative' in available_classes:
        balanced_dfs.append(resample(df_neg, replace=False, n_samples=min_size, random_state=42))
    if 'neutral' in available_classes:
        balanced_dfs.append(resample(df_neu, replace=False, n_samples=min_size, random_state=42))
    if 'positive' in available_classes:
        balanced_dfs.append(resample(df_pos, replace=False, n_samples=min_size, random_state=42))

    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
balanced_df.to_csv(output_file, index=False)
print(f"\n✅ Balanced dataset saved to: {output_file}")

print("\n📊 Final Balanced Class Distribution:")
print(balanced_df['sentiment'].value_counts())