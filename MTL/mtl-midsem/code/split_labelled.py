import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "results", "processed_csv", "labelled_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "processed_csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)
print(f"Total rows in labelled dataset: {len(df)}")

# Step 1: Split off 70% for SSL pretraining (unlabelled usage simulation)
ssl_data, remaining = train_test_split(df, test_size=0.30, stratify=df["emotion"], random_state=42)

# Step 2: From the remaining 30%, split into 20% general classifier, 10% fine-tuning
classifier_data, finetune_data = train_test_split(
    remaining,
    test_size=0.3333,  # 10% out of total = 1/3 of the 30%
    stratify=remaining["emotion"],
    random_state=42
)

# Save datasets
ssl_path = os.path.join(OUTPUT_DIR, "ssl_dataset.csv")
classifier_path = os.path.join(OUTPUT_DIR, "classifier_dataset.csv")
finetune_path = os.path.join(OUTPUT_DIR, "finetune_dataset.csv")

ssl_data.to_csv(ssl_path, index=False)
classifier_data.to_csv(classifier_path, index=False)
finetune_data.to_csv(finetune_path, index=False)

# Print summary
print(f"SSL dataset rows (70%): {len(ssl_data)}")
print(f"Classifier dataset rows (20%): {len(classifier_data)}")
print(f"Fine-tuning dataset rows (10%): {len(finetune_data)}")

print("\nPer-emotion counts (SSL):")
print(ssl_data["emotion"].value_counts())

print("\nPer-emotion counts (Classifier):")
print(classifier_data["emotion"].value_counts())

print("\nPer-emotion counts (Fine-tuning):")
print(finetune_data["emotion"].value_counts())

print(f"\nSaved SSL dataset to: {ssl_path}")
print(f"Saved Classifier dataset to: {classifier_path}")
print(f"Saved Fine-tuning dataset to: {finetune_path}")
