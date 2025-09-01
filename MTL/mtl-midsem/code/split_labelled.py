# split_labelled.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# -----------------------------
# Paths relative to this script
# -----------------------------
base_dir = Path(__file__).resolve().parent.parent  # repo root
labelled_path = base_dir / "results" / "processed_csv" / "labelled_dataset.csv"
fine_tune_path = base_dir / "results" / "processed_csv" / "fine_tune_dataset.csv"
test_path = base_dir / "results" / "processed_csv" / "test_dataset.csv"

# -----------------------------
# Load labelled dataset
# -----------------------------
if not labelled_path.exists():
    raise FileNotFoundError(f"Labelled dataset not found at {labelled_path}")

df = pd.read_csv(labelled_path)

# -----------------------------
# Stratified split by 'emotion'
# -----------------------------
fine_tune_df, test_df = train_test_split(
    df,
    test_size=0.2,       # 20% for testing
    random_state=42,
    stratify=df['emotion']
)

# -----------------------------
# Save outputs
# -----------------------------
fine_tune_df.to_csv(fine_tune_path, index=False)
test_df.to_csv(test_path, index=False)

print("Fine-tuning dataset rows:", len(fine_tune_df))
print("Test dataset rows:", len(test_df))
print("\nPer-emotion counts (Fine-tuning):\n", fine_tune_df['emotion'].value_counts())
print("\nPer-emotion counts (Test):\n", test_df['emotion'].value_counts())
print(f"\nSaved fine-tune dataset to: {fine_tune_path}")
print(f"Saved test dataset to: {test_path}")
