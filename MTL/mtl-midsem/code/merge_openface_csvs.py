import pandas as pd, os

base = "/Users/niha/mtl-emotion/features/faces/openface_csv"
classes = ["Anger","Happy","Neutral","Sad"]

dfs = []
for cls in classes:
    csv_path = os.path.join(base, cls, f"{cls}.csv")
    if not os.path.exists(csv_path):
        print("⚠ missing", csv_path)
        continue
    df = pd.read_csv(csv_path)
    df["label"] = cls
    dfs.append(df)

out = "features/faces/affectnet_openface.parquet"
pd.concat(dfs, ignore_index=True).to_parquet(out)
print("✅ merged CSVs →", out)
