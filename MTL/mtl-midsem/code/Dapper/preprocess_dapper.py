# === FILE: code/Dapper/preprocess_dapper.py ===
"""
Preprocess DAPPER physiological data safely:

- Load raw CSVs (HR, GSR, PPG, ACC)
- Convert timestamps to seconds
- Create sliding-window features
- Attach optional labels if available
- Save processed parquet file incrementally
"""

import os
import glob
import pandas as pd
import numpy as np
import yaml
from features import sliding_windows, extract_window_features

def load_config(path='code/Dapper/config.yaml'):
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def process_participant(part_dir, cfg):
    """Process all CSV recordings of a participant into windowed features."""
    csv_files = [f for f in os.listdir(part_dir) 
                 if f.endswith('.csv') and not f.endswith(('_ACC.csv', '_GSR.csv', '_PPG.csv'))]
    records = []

    for csv_file in csv_files:
        csv_path = os.path.join(part_dir, csv_file)
        if os.path.getsize(csv_path) == 0:
            print(f"⚠️  Skipping empty file: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️  Skipping unreadable file: {csv_path}")
            continue

        # --- Handle timestamps ---
        if 'timestamp' in df.columns:
            try:
                t = pd.to_datetime(df['timestamp']).astype('int64') // 1_000_000_000
                df['t_sec'] = t - t.iloc[0]
            except Exception:
                df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']
        else:
            df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']

        timestamps = df['t_sec'].values

        # --- Sensors (fallback if missing) ---
        sensors = {}
        for col in ['hr', 'gsr', 'acc_x', 'acc_y', 'acc_z']:
            if col in df.columns:
                sensors[col] = df[col].fillna(method='ffill').fillna(0).values
            else:
                sensors[col] = np.zeros(len(df))

        # --- Reduced windowing ---
        window_s = cfg['features']['window_seconds']   # e.g., 30 sec
        step_s = cfg['features']['step_seconds'] * 2  # doubled step to reduce overlap
        fs = cfg['features']['sampling_rate']

        # --- Sliding windows ---
        hr_windows = sliding_windows(timestamps, sensors['hr'], window_s, step_s, fs)
        gsr_windows = sliding_windows(timestamps, sensors['gsr'], window_s, step_s, fs)
        acc_x_windows = sliding_windows(timestamps, sensors['acc_x'], window_s, step_s, fs)
        acc_y_windows = sliding_windows(timestamps, sensors['acc_y'], window_s, step_s, fs)
        acc_z_windows = sliding_windows(timestamps, sensors['acc_z'], window_s, step_s, fs)

        n_windows = max(len(hr_windows), len(gsr_windows), len(acc_x_windows))
        for i in range(n_windows):
            window_signals = {
                'hr': hr_windows[i] if i < len(hr_windows) else np.array([]),
                'gsr': gsr_windows[i] if i < len(gsr_windows) else np.array([]),
                'acc_x': acc_x_windows[i] if i < len(acc_x_windows) else np.array([]),
                'acc_y': acc_y_windows[i] if i < len(acc_y_windows) else np.array([]),
                'acc_z': acc_z_windows[i] if i < len(acc_z_windows) else np.array([]),
            }

            feats = extract_window_features(timestamps, window_signals, fs)

            center_t = (i * step_s) + (window_s / 2)
            rec = {
                'participant_id': os.path.basename(part_dir),
                'window_id': i,
                'start_time': i * step_s,
                'center_time': center_t
            }
            rec.update(feats)

            # Optional labels if present in the CSV
            for lab in ['valence', 'arousal', 'panas_pos', 'panas_neg']:
                if lab in df.columns:
                    label_row = df.iloc[(np.abs(df['t_sec'] - center_t)).argmin()]
                    rec[lab] = label_row.get(lab, np.nan)

            records.append(rec)

    return records

def main(raw_root=None, out_path=None):
    cfg = load_config()

    raw_root = raw_root or "code/Dapper/dataset_files/Physiol_Rec1/Physiol_Rec"
    out_path = out_path or "code/Dapper/data/processed/dapper_features.parquet"

    participants = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]

    # Process each participant separately and append to parquet to save memory
    for idx, pid in enumerate(participants):
        part_dir = os.path.join(raw_root, pid)
        print(f"Processing participant {pid} ({idx+1}/{len(participants)})")
        recs = process_participant(part_dir, cfg)
        df_part = pd.DataFrame(recs)

        # Incremental saving
        if not os.path.exists(out_path):
            df_part.to_parquet(out_path, index=False)
        else:
            df_existing = pd.read_parquet(out_path)
            df_combined = pd.concat([df_existing, df_part], ignore_index=True)
            df_combined.to_parquet(out_path, index=False)

    print(f"✅ Saved processed features to {out_path}")

if __name__ == '__main__':
    main()
