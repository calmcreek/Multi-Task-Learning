# === FILE: code/Dapper/split_dapper.py ===
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load the processed parquet
data_path = "code/Dapper/data/processed/dapper_features.parquet"
df = pd.read_parquet(data_path)

# Each row should already include a subject_id column (if not, add during preprocessing!)
if "participant_id" not in df.columns:
    raise ValueError("participant_id column missing! Ensure preprocess_dapper.py saves it.")

print("Number of unique participants in processed data:", df['participant_id'].nunique())

# Unique participants
subjects = df["participant_id"].unique()
train_subj, test_subj = train_test_split(subjects, test_size=0.15, random_state=42)
train_subj, val_subj = train_test_split(train_subj, test_size=0.15, random_state=42)

# Create splits
train_df = df[df["participant_id"].isin(train_subj)]
val_df   = df[df["participant_id"].isin(val_subj)]
test_df  = df[df["participant_id"].isin(test_subj)]

# Save to disk
os.makedirs("code/Dapper/data/splits", exist_ok=True)
train_df.to_parquet("code/Dapper/data/splits/train.parquet")
val_df.to_parquet("code/Dapper/data/splits/val.parquet")
test_df.to_parquet("code/Dapper/data/splits/test.parquet")

print(f"Saved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
