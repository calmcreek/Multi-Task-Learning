# === FILE: code/Dapper/features.py ===
import numpy as np
import pandas as pd
from scipy import signal, stats
import warnings

def safe_moments(x):
    """Compute mean, std, skew, kurtosis safely, ignoring RuntimeWarnings."""
    if x.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurtosis': np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        skew = stats.skew(x) if np.std(x) > 1e-8 else 0.0
        kurt = stats.kurtosis(x) if np.std(x) > 1e-8 else 0.0
    return {
        'mean': np.mean(x),
        'std': np.std(x),
        'skew': skew,
        'kurtosis': kurt
    }

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
        ws.append(values[idx] if idx.any() else np.array([]))
        t += step_s
    return ws

def time_domain_features(x):
    """Compute robust time-domain features for a window."""
    if x.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'skew': np.nan, 'kurtosis': np.nan, 'median': np.nan, 'iqr': np.nan
        }
    moments = safe_moments(x)
    iqr = np.nanpercentile(x,75) - np.nanpercentile(x,25)
    median = np.nanmedian(x)
    return {**moments, 'median': median, 'iqr': iqr, 'min': np.nanmin(x), 'max': np.nanmax(x)}

def spectral_features(x, fs):
    """Compute simple spectral features using Welch's method."""
    if x.size < 4:
        return {'pwr_total': np.nan, 'pwr_band_1': np.nan, 'pwr_band_2': np.nan}
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    pwr_total = np.trapz(Pxx, f)
    # example bands (adjust if needed)
    b1 = ((f >= 0.04) & (f < 0.15))
    b2 = ((f >= 0.15) & (f < 0.4))
    return {
        'pwr_total': pwr_total,
        'pwr_band_1': np.trapz(Pxx[b1], f[b1]) if b1.any() else 0.0,
        'pwr_band_2': np.trapz(Pxx[b2], f[b2]) if b2.any() else 0.0
    }

def extract_window_features(timestamps, signal_array, fs):
    """
    signal_array: dict of modality -> 1d numpy array aligned with timestamps
    Returns dict of all features with prefixed keys.
    """
    feat = {}
    for name, arr in signal_array.items():
        td = time_domain_features(arr)
        sp = spectral_features(arr, fs)
        # Combine features with prefix
        for k, v in {**td, **sp}.items():
            feat[f"{name}_{k}"] = v
    return feat
