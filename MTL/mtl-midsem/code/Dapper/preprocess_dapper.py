import pandas as pd
import numpy as np
import os
from dataset import DapperDataset

raw_dir = "code/Dapper/dataset_files/Pre-test"
out_path = "code/Dapper/data/processed/dapper_features.parquet"

df = pd.read_excel(os.path.join(raw_dir, "trait.xls"))

# Create dummy features for testing
n = len(df)
df_dummy = pd.DataFrame({
    "hr_mean": np.random.rand(n),
    "gsr_mean": np.random.rand(n),
    "acc_x_mean": np.random.rand(n),
    "acc_y_mean": np.random.rand(n),
    "acc_z_mean": np.random.rand(n),
    "valence": df.get("valence", np.random.rand(n)),
    "arousal": df.get("arousal", np.random.rand(n)),
    "panas_pos": df.get("positive_affect", np.random.rand(n)),
})
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_dummy.to_parquet(out_path, index=False)
print("Saved dummy features for testing at", out_path)


# # === FILE: code/Dapper/preprocess_dapper.py ===
# """
# Preprocess DAPPER (or test) data:
# - Load raw CSVs
# - Convert timestamps or fallback to row-index time
# - Create sliding-window features
# - Save processed parquet file
# """

# import os
# import glob
# import pandas as pd
# import numpy as np
# import yaml
# from features import sliding_windows, extract_window_features


# def load_config(path='code/Dapper/config.yaml'):
#     """Load YAML config."""
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)


# def process_participant(csv_path, cfg):
#     """Process one participant CSV into windowed features."""
#     df = pd.read_csv(csv_path)

#     # --- Handle timestamp ---
#     if 'timestamp' in df.columns:
#         try:
#             t = pd.to_datetime(df['timestamp']).astype('int64') // 1_000_000_000
#             df['t_sec'] = t - t.iloc[0]
#         except Exception:
#             df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']
#     else:
#         df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']

#     timestamps = df['t_sec'].values

#     # --- Sensors (handle missing cols gracefully) ---
#     sensors = {}
#     for col in ['hr', 'gsr', 'acc_x', 'acc_y', 'acc_z']:
#         if col in df.columns:
#             sensors[col] = df[col].fillna(method='ffill').fillna(0).values
#         else:
#             sensors[col] = np.zeros(len(df))  # fallback

#     window_s = cfg['features']['window_seconds']
#     step_s = cfg['features']['step_seconds']
#     fs = cfg['features']['sampling_rate']

#     # --- Sliding windows ---
#     hr_windows = sliding_windows(timestamps, sensors['hr'], window_s, step_s, fs)
#     gsr_windows = sliding_windows(timestamps, sensors['gsr'], window_s, step_s, fs)
#     acc_x_windows = sliding_windows(timestamps, sensors['acc_x'], window_s, step_s, fs)
#     acc_y_windows = sliding_windows(timestamps, sensors['acc_y'], window_s, step_s, fs)
#     acc_z_windows = sliding_windows(timestamps, sensors['acc_z'], window_s, step_s, fs)

#     n = max(len(hr_windows), len(gsr_windows), len(acc_x_windows))
#     records = []

#     for i in range(n):
#         window_signals = {
#             'hr': hr_windows[i] if i < len(hr_windows) else np.array([]),
#             'gsr': gsr_windows[i] if i < len(gsr_windows) else np.array([]),
#             'acc_x': acc_x_windows[i] if i < len(acc_x_windows) else np.array([]),
#             'acc_y': acc_y_windows[i] if i < len(acc_y_windows) else np.array([]),
#             'acc_z': acc_z_windows[i] if i < len(acc_z_windows) else np.array([]),
#         }

#         feats = extract_window_features(timestamps, window_signals, fs)

#         # --- Labels (optional, may not exist in test data) ---
#         center_t = (i * step_s) + (window_s / 2)
#         label_row = (
#             df.iloc[(np.abs(df['t_sec'] - center_t)).argmin()]
#             if 'valence' in df.columns
#             else None
#         )

#         rec = {
#             'participant_id': os.path.basename(csv_path).split('.')[0],
#             'window_id': i,
#             'start_time': i * step_s,
#             'center_time': center_t,
#         }
#         rec.update(feats)

#         if label_row is not None:
#             for lab in ['valence', 'arousal', 'panas_pos', 'panas_neg']:
#                 if lab in df.columns:
#                     rec[lab] = label_row.get(lab, np.nan)

#         records.append(rec)

#     return records


# def main(raw_dir=None, out_path=None):
#     cfg = load_config()

#     # ✅ For testing, point raw_dir to ./Pre-test
#     raw_dir = raw_dir or "./Pre-test"
#     out_path = out_path or "outputs/test_features.parquet"

#     all_records = []
#     csvs = glob.glob(os.path.join(raw_dir, '*.csv'))
#     if not csvs:
#         print(f"No CSVs found in {raw_dir}")
#         return

#     for p in csvs:
#         print('Processing', p)
#         recs = process_participant(p, cfg)
#         all_records.extend(recs)

#     df_out = pd.DataFrame(all_records)
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     df_out.to_parquet(out_path, index=False)
#     print('✅ Saved features to', out_path)


# if __name__ == '__main__':
#     main()
