# === FILE: code/Dapper/features.py ===
import numpy as np
import pandas as pd
from scipy import signal, stats


def sliding_windows(timestamps, values, window_s, step_s, fs):
    """Return windows as list of arrays. timestamps in seconds."""
    if len(timestamps) == 0:
        return []
    start = timestamps[0]
    end = timestamps[-1]
    ws = []
    t = start
    while t + window_s <= end:
        idx = (timestamps >= t) & (timestamps < t + window_s)
        if idx.any():
            ws.append(values[idx])
        else:
            ws.append(np.array([]))
        t += step_s
    return ws


def time_domain_features(x):
    if x.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'skew': np.nan, 'kurtosis': np.nan, 'median': np.nan, 'iqr': np.nan
        }
    return {
        'mean': np.nanmean(x),
        'std': np.nanstd(x),
        'min': np.nanmin(x),
        'max': np.nanmax(x),
        'skew': stats.skew(x),
        'kurtosis': stats.kurtosis(x),
        'median': np.nanmedian(x),
        'iqr': np.nanpercentile(x,75) - np.nanpercentile(x,25)
    }


def spectral_features(x, fs):
    if x.size < 4:
        return {'pwr_total': np.nan, 'pwr_band_1': np.nan, 'pwr_band_2': np.nan}
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    pwr_total = np.trapz(Pxx, f)
    # example bands (user adjust)
    b1 = ((f>=0.04)&(f<0.15))
    b2 = ((f>=0.15)&(f<0.4))
    return {
        'pwr_total': pwr_total,
        'pwr_band_1': np.trapz(Pxx[b1], f[b1]) if b1.any() else 0.0,
        'pwr_band_2': np.trapz(Pxx[b2], f[b2]) if b2.any() else 0.0
    }


def extract_window_features(timestamps, signal_array, fs):
    """signal_array: dict of modality->1d numpy array aligned with timestamps"""
    feat = {}
    # HR/PPG/ECG like signals
    for name, arr in signal_array.items():
        td = time_domain_features(arr)
        sp = spectral_features(arr, fs)
        # prefix
        for k,v in {**td, **sp}.items():
            feat[f"{name}_{k}"] = v
    return feat
