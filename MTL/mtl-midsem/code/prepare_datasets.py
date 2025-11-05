import os
import sys
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path

# -----------------------------
# CONFIG: paths
# -----------------------------
# BASE_DIR points to repo root (mtl-midsem/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Input CSVs for each emotion
# Corrected input and output paths
CSV_ROOT = BASE_DIR / "data" / "affectnet" / "features"

emotion_files = {
    "angry": CSV_ROOT / "anger" / "anger.csv",
    "sad": CSV_ROOT / "sad" / "sad.csv",
    "happy": CSV_ROOT / "happy" / "happy.csv",
    "neutral": CSV_ROOT / "neutral" / "neutral.csv",
    "fear": CSV_ROOT / "fear" / "fear.csv",
    "disgust": CSV_ROOT / "disgust" / "disgust.csv",
    "contempt": CSV_ROOT / "contempt" / "contempt.csv",
    "surprise": CSV_ROOT / "surprise" / "surprise.csv",
}

OUT_DIR = BASE_DIR / "mtl-midsem" / "results" / "processed_csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
pretrain_out = OUT_DIR / "pretrain_dataset.csv"
labelled_out = OUT_DIR / "labelled_dataset.csv"

# Split ratio
split_ratio = 0.7
random_seed = 42

# Columns that may look like labels, we remove from pretrain
possible_label_cols = ['emotion', 'label', 'class', 'target', 'stress', 'valence', 'arousal']

# -----------------------------
# Helper: add labels for labelled portion
# -----------------------------
def add_labels(df, emotion):
    for c in ['emotion', 'stress', 'valence', 'arousal']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.copy()
    df['emotion'] = emotion
    df['stress'] = 1 if emotion in ("angry", "sad") else 0
    mapping = {
    "happy": (2.0, 2.0),
    "sad": (1.0, 1.0),
    "angry": (1.0, 2.0),
    "neutral": (1.5, 1.5),
    "fear": (0.8, 2.5),
    "disgust": (0.9, 2.0),
    "contempt": (1.2, 1.8),
    "surprise": (2.2, 2.5),
    }

    val, aro = mapping[emotion]
    df['valence'] = val
    df['arousal'] = aro
    return df

# -----------------------------
# Main processing
# -----------------------------
pretrain_parts = []
labelled_parts = []
summary = {}

for emotion, path in emotion_files.items():
    print(f"\n--- Processing '{emotion}' from: {path}")
    if not path.exists():
        print(f"ERROR: file not found -> {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    n = len(df)
    print(f"Rows found: {n}")
    if n == 0:
        continue

    df = shuffle(df, random_state=random_seed).reset_index(drop=True)
    split_idx = int(n * split_ratio)
    df_pre = df.iloc[:split_idx].copy()
    df_lab = df.iloc[split_idx:].copy()

    # drop labels from pretrain
    drop_cols = [c for c in possible_label_cols if c in df_pre.columns]
    if drop_cols:
        df_pre = df_pre.drop(columns=drop_cols)

    # add labels to labelled
    df_lab = add_labels(df_lab, emotion)

    pretrain_parts.append(df_pre)
    labelled_parts.append(df_lab)

    summary[emotion] = {"total": n, "pretrain": len(df_pre), "labelled": len(df_lab)}

# merge and save
pretrain_df = pd.concat(pretrain_parts, ignore_index=True)
labelled_df = pd.concat(labelled_parts, ignore_index=True)

pretrain_df.to_csv(pretrain_out, index=False)
labelled_df.to_csv(labelled_out, index=False)

print("\n=== DONE ===")
print(f"Saved {pretrain_out} with {len(pretrain_df)} rows")
print(f"Saved {labelled_out} with {len(labelled_df)} rows")

print("\nPer-emotion split summary:")
for emo, stats in summary.items():
    tot = stats['total']
    p = stats['pretrain']
    l = stats['labelled']
    print(f"  {emo:7s}: total={tot:6d} | pretrain={p:6d} | labelled={l:6d}")
