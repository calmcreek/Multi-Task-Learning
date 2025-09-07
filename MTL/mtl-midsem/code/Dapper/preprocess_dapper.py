"""
Preprocess DAPPER physiological + psychological data safely:

- Load raw CSVs (HR, GSR, PPG, ACC)
- Detect correct column names dynamically
- Convert timestamps to seconds
- Create sliding-window features
- Attach labels from DRM + ESM
- Save processed parquet incrementally (PyArrow writer)
"""

import os
import pandas as pd
import numpy as np
import yaml
from features import sliding_windows, extract_window_features

import warnings
from scipy import stats
import pyarrow as pa
import pyarrow.parquet as pq


def safe_moments(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            'mean': np.mean(x),
            'std': np.std(x),
            'skew': stats.skew(x),
            'kurtosis': stats.kurtosis(x)
        }


# -------------------------------
# CONFIG
# -------------------------------
def load_config(path='code/Dapper/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# -------------------------------
# PSYCHOL DATA (DRM + ESM)
# -------------------------------
def load_psychol_data(drm_path, esm_path):
    drm = pd.read_excel(drm_path)
    esm = pd.read_excel(esm_path)

    # Try to detect timestamp columns dynamically
    drm_time_col = next((c for c in drm.columns if "time" in c.lower()), None)
    esm_time_col = next((c for c in esm.columns if "time" in c.lower()), None)

    if drm_time_col is None or esm_time_col is None:
        raise ValueError(
            f"❌ Could not detect timestamp columns in DRM/ESM. DRM cols={drm.columns}, ESM cols={esm.columns}"
        )

    drm['timestamp'] = pd.to_datetime(drm[drm_time_col])
    esm['timestamp'] = pd.to_datetime(esm[esm_time_col])

    # PANAS aggregation
    panas_pos_items = [c for c in drm.columns if str(c).startswith("PANAS_") and int(c.split("_")[1]) in [1, 3, 5, 9]]
    panas_neg_items = [c for c in drm.columns if str(c).startswith("PANAS_") and int(c.split("_")[1]) in [2, 4, 6, 8, 10]]

    for df in [drm, esm]:
        if panas_pos_items:
            df['panas_pos'] = df[panas_pos_items].mean(axis=1)
        if panas_neg_items:
            df['panas_neg'] = df[panas_neg_items].mean(axis=1)

    # Keep consistent columns
    keep_cols = ['Participant ID', 'timestamp', 'Valence', 'Arousal', 'panas_pos', 'panas_neg']
    drm = drm[[c for c in keep_cols if c in drm.columns]]
    esm = esm[[c for c in keep_cols if c in esm.columns]]

    return pd.concat([drm, esm], ignore_index=True)


# -------------------------------
# PHYSIOL DATA
# -------------------------------
def normalize_sensor_columns(df):
    """Map various sensor file columns into unified names: hr, gsr, acc_x, acc_y, acc_z"""
    col_map = {}

    for c in df.columns:
        if "heart" in c.lower() or c.lower().startswith("hr"):
            col_map['hr'] = c
        if c.lower().startswith("gsr"):
            col_map['gsr'] = c
        if c.lower().startswith("ppg"):
            col_map['ppg'] = c
        if "motion_datax" in c.lower() or c.lower() == "motionx":
            col_map['acc_x'] = c
        if "motion_datay" in c.lower() or c.lower() == "motiony":
            col_map['acc_y'] = c
        if "motion_dataz" in c.lower() or c.lower() == "motionz":
            col_map['acc_z'] = c

    return col_map


def detect_time_column(df):
    """Find a timestamp column dynamically"""
    for c in df.columns:
        if "time" in c.lower():
            return c
    return None


def process_participant(part_dir, cfg, psychol_df):
    records = []

    csv_files = [f for f in os.listdir(part_dir) if f.endswith(".csv")]
    for csv_file in csv_files:
        csv_path = os.path.join(part_dir, csv_file)
        if os.path.getsize(csv_path) == 0:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            print(f"⚠️ Could not read file: {csv_path}")
            continue

        # Detect time column
        time_col = detect_time_column(df)
        if time_col:
            try:
                t = pd.to_datetime(df[time_col]).astype("int64") // 1_000_000_000
                df['t_sec'] = t - t.iloc[0]
            except Exception:
                df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']
        else:
            df['t_sec'] = np.arange(len(df)) / cfg['features']['sampling_rate']

        timestamps = df['t_sec'].values

        # Normalize sensor columns (fallback to default names)
        col_map = {
            'hr': 'heart_rate',
            'gsr': 'GSR',
            'acc_x': 'Motion_dataX',
            'acc_y': 'Motion_dataY',
            'acc_z': 'Motion_dataZ'
        }
        sensors = {}
        for name in ['hr', 'gsr', 'acc_x', 'acc_y', 'acc_z']:
            if col_map[name] in df.columns:
                sensors[name] = pd.to_numeric(df[col_map[name]], errors='coerce').ffill().fillna(0).values
            else:
                sensors[name] = np.zeros(len(df))

        # Sliding windows
        window_s = cfg['features']['window_seconds']
        step_s = cfg['features']['step_seconds']
        fs = cfg['features']['sampling_rate']

        sig_windows = {k: sliding_windows(timestamps, v, window_s, step_s, fs) for k, v in sensors.items()}
        n_windows = max(len(w) for w in sig_windows.values())

        pid = int(os.path.basename(part_dir))

        for i in range(n_windows):
            window_signals = {k: sig_windows[k][i] if i < len(sig_windows[k]) else np.array([]) for k in sig_windows}
            feats = extract_window_features(timestamps, window_signals, fs)

            center_t = (i * step_s) + (window_s / 2)
            rec = {
                'participant_id': pid,
                'window_id': i,
                'start_time': i * step_s,
                'center_time': center_t
            }
            rec.update(feats)

            # Attach psychol labels
            sub_df = psychol_df[psychol_df['Participant ID'] == pid]
            if not sub_df.empty:
                target_time = sub_df['timestamp'].iloc[0] + pd.to_timedelta(center_t, unit='s')
                nearest_idx = (sub_df['timestamp'] - target_time).abs().idxmin()
                nearest = sub_df.loc[nearest_idx]
                for lab in ['Valence', 'Arousal', 'panas_pos', 'panas_neg']:
                    rec[lab.lower()] = nearest.get(lab, np.nan)
            else:
                rec.update({'valence': np.nan, 'arousal': np.nan, 'panas_pos': np.nan, 'panas_neg': np.nan})

            records.append(rec)

    return records


# -------------------------------
# MAIN
# -------------------------------
def main(raw_root=None, out_path=None):
    cfg = load_config()
    raw_root = raw_root or "code/Dapper/dataset_files/Physiol_Rec1/Physiol_Rec"
    out_path = out_path or "code/Dapper/data/processed/dapper_features.parquet"

    psychol_df = load_psychol_data(
        "code/Dapper/dataset_files/Psychol_Rec/DRM.xlsx",
        "code/Dapper/dataset_files/Psychol_Rec/ESM.xlsx"
    )

    participants = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer = None
    for idx, pid in enumerate(participants):
        print(f"Processing participant {pid} ({idx+1}/{len(participants)})")
        recs = process_participant(os.path.join(raw_root, pid), cfg, psychol_df)
        df_part = pd.DataFrame(recs)

        if df_part.empty:
            continue

        table = pa.Table.from_pandas(df_part, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"✅ Incremental saving complete → {out_path}")


if __name__ == '__main__':
    main()
