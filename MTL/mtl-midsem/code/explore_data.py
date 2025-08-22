import pandas as pd

df = pd.read_parquet("features/faces/affectnet_openface.parquet")
print("Shape:", df.shape)
print("Columns:", df.columns[:15])
print("Class distribution:\n", df["label"].value_counts())
