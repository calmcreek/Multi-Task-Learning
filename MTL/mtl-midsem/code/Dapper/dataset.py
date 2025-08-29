# === FILE: code/Dapper/dataset.py ===
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class DapperDataset(Dataset):
    def __init__(self, parquet_path, feature_cols=None, task_cols=None, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.feature_cols = feature_cols or [c for c in self.df.columns if c.startswith(('hr_','gsr_','acc_'))]
        self.task_cols = task_cols or []   # ← leave empty if nothing passed
        self.transform = transform

        # drop rows with all-nan features
        self.df = self.df.dropna(axis=0, how='all', subset=self.feature_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_cols].fillna(0).values.astype(np.float32))
        y = {}
        mask = {}
        for t in self.task_cols:
            if t in row and not pd.isna(row[t]):
                y[t] = torch.tensor(float(row[t]), dtype=torch.float32)
                mask[t] = torch.tensor(1.0)
            else:
                y[t] = torch.tensor(0.0)
                mask[t] = torch.tensor(0.0)
        return x, y, mask
