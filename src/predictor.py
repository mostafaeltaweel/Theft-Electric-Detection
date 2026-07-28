"""
Prediction & Feature Pipeline Engine for ETD-XAI.
Preserves exact training feature extraction (59 statistical features),
min-max sequence scaling, and CNN-LSTM inference.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

from src.config import SCALER_PATH, DEFAULT_THRESHOLD, N_STAT_FEATURES
from src.model_loader import ENGINE, is_model_loaded, get_tf

def scale_sequences(readings: np.ndarray) -> np.ndarray:
    """Per-row min-max scale each sequence to [0,1] — matches training CELL 8."""
    readings = np.asarray(readings, dtype=np.float32)
    scaled = np.zeros_like(readings)
    for i in range(len(readings)):
        mn, mx = readings[i].min(), readings[i].max()
        if mx > mn:
            scaled[i] = (readings[i] - mn) / (mx - mn)
    return scaled

def _features_for_row(row: np.ndarray) -> list:
    """59 statistical features verbatim from training CELL 7."""
    row = row.astype(np.float32)
    n = len(row)
    mean = np.mean(row); std = np.std(row); mx = np.max(row); mn = np.min(row)
    median = np.median(row)
    skew = float(scipy_stats.skew(row)); kurt = float(scipy_stats.kurtosis(row))
    cv = std / (mean + 1e-9)
    p10, p25, p75, p90 = np.percentile(row, [10, 25, 75, 90])
    iqr = p75 - p25
    zero_ratio = np.mean(row == 0); neg_ratio = np.mean(row < 0)
    near_zero = np.mean(row < 0.01); low_cons_ratio = np.mean(row < mean * 0.1)
    diffs_tmp = np.diff(row)
    drop_ratio = np.mean(diffs_tmp < -std) if len(diffs_tmp) > 0 else 0.0
    t = np.arange(n)
    slope = np.polyfit(t, row, 1)[0]
    resid = row - np.polyval(np.polyfit(t, row, 1), t)
    resid_std = np.std(resid)
    energy = np.sum(row ** 2) / n
    hist, _ = np.histogram(row, bins=min(30, n), density=True)
    ent = entropy(hist + 1e-9)
    runs, cnt = [], 0
    for v in row:
        if v == 0:
            cnt += 1
        else:
            if cnt > 0: runs.append(cnt)
            cnt = 0
    if cnt > 0: runs.append(cnt)
    max_zero_run = max(runs) if runs else 0
    n_zero_runs = len(runs)
    if n >= 7:
        day_chg = np.abs(np.diff(row))
        max_day_chg = np.max(day_chg) if len(day_chg) > 0 else 0
        mean_day_chg = np.mean(day_chg) if len(day_chg) > 0 else 0
        dm_mean = np.mean(row); dm_std = np.std(row)
        dm_max = np.max(row); dm_min = np.min(row); ds_mean = np.std(row)
        day_cv = dm_std / (dm_mean + 1e-9)
        theft_days = np.mean(row < dm_mean * 0.5)
        week1 = row[:7] if n >= 14 else row[:n // 2]
        week2 = row[7:14] if n >= 14 else row[n // 2:]
        dn_ratio = np.mean(week1) / (np.mean(week2) + 1e-9)
    else:
        dn_ratio = day_cv = theft_days = 0
        max_day_chg = mean_day_chg = 0
        dm_mean = dm_std = dm_max = dm_min = ds_mean = 0
    ac1 = float(np.corrcoef(row[:-1], row[1:])[0, 1]) if n > 1 else 0.0
    ac48 = float(np.corrcoef(row[:-7], row[7:])[0, 1]) if n > 7 else 0.0
    ac7d = float(np.corrcoef(row[:-14], row[14:])[0, 1]) if n > 14 else 0.0
    fft_v = np.abs(np.fft.rfft(row))
    fft_mean = np.mean(fft_v); fft_std = np.std(fft_v); fft_max = np.max(fft_v)
    dominant_freq = np.argmax(fft_v[1:]) + 1
    if n >= 14:
        mean_change = np.mean(row[n // 2:]) - np.mean(row[:n // 2])
        std_change = np.std(row[n // 2:]) - np.mean(row[:n // 2])
    else:
        mean_change = std_change = 0.0
    diffs = np.diff(row)
    max_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    max_rise = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    n_big_drops = int(np.sum(diffs < -2 * std)); n_big_rises = int(np.sum(diffs > 2 * std))
    below_median = np.mean(row < median); above_median = np.mean(row > median)
    quarters = np.array_split(row, 4)
    q_means = [np.mean(q) for q in quarters]; q_stds = [np.std(q) for q in quarters]
    q_trend = q_means[-1] - q_means[0]; q_var = np.std(q_means)
    return [mean, std, mx, mn, median, skew, kurt, cv, p10, p25, p75, p90, iqr,
            zero_ratio, neg_ratio, near_zero, low_cons_ratio, drop_ratio,
            slope, resid_std, energy, ent, max_zero_run, n_zero_runs,
            dn_ratio, day_cv, theft_days, max_day_chg, mean_day_chg,
            dm_mean, dm_std, dm_max, dm_min, ds_mean, ac1, ac48, ac7d,
            fft_mean, fft_std, fft_max, dominant_freq, mean_change, std_change,
            max_drop, max_rise, n_big_drops, n_big_rises, below_median, above_median,
            q_means[0], q_means[1], q_means[2], q_means[3],
            q_stds[0], q_stds[1], q_stds[2], q_stds[3], q_trend, q_var]

def extract_features(readings: np.ndarray) -> np.ndarray:
    feats = np.array([_features_for_row(r) for r in readings], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

class FeaturePipeline:
    """Scaler pipeline using saved training stat_scaler.pkl."""
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._locked = False
        self._load_saved()

    def _load_saved(self):
        if SCALER_PATH.exists():
            try:
                import joblib
                sc = joblib.load(SCALER_PATH)
                if hasattr(sc, "transform"):
                    self._scaler = sc
                    self._fitted = True
                    self._locked = True
            except Exception:
                self._scaler = None; self._fitted = False; self._locked = False

    def transform(self, readings: np.ndarray) -> np.ndarray:
        raw = extract_features(readings)
        if self._fitted and self._scaler is not None:
            out = self._scaler.transform(raw).astype(np.float32)
        else:
            self._scaler = StandardScaler()
            out = self._scaler.fit_transform(raw).astype(np.float32)
            self._fitted = True
        return np.nan_to_num(out)

PIPELINE = FeaturePipeline()

def resize_sequences(seq_2d: np.ndarray, target_len: int = 120) -> np.ndarray:
    """Resizes 2D sequence matrix to target_len (120) by truncating/padding."""
    seq_2d = np.asarray(seq_2d, dtype=np.float32)
    if seq_2d.shape[1] == target_len:
        return seq_2d
    rows = []
    for r in seq_2d:
        L = len(r)
        if L >= target_len:
            rows.append(r[:target_len])
        else:
            rows.append(np.concatenate([r, np.zeros(target_len - L, dtype=np.float32)]))
    return np.vstack(rows).astype(np.float32)

def classify(prob: float, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Classifies risk probability score into status label."""
    pred = 1 if prob >= threshold else 0
    conf = prob if pred == 1 else (1.0 - prob)
    risk = round(prob * 100, 2)
    level = "High" if risk >= 75 else "Medium" if risk >= 40 else "Low"
    return {
        "probability": round(float(prob), 6),
        "prediction": pred,
        "confidence": round(float(conf), 6),
        "risk_score": risk,
        "risk_level": level,
        "status": "Theft" if pred == 1 else "Normal"
    }

def predict_sequences(raw_2d: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Core CNN-LSTM model inference engine."""
    if not is_model_loaded() or ENGINE.model is None:
        raise RuntimeError("No active CNN-LSTM model loaded.")
        
    raw_2d = np.asarray(raw_2d, dtype=np.float32)
    ready = resize_sequences(raw_2d, 120)
    seq_scaled = scale_sequences(ready)
    
    seq_in = seq_scaled.reshape(-1, 120, 1).astype(np.float32)
    stat_in = PIPELINE.transform(ready).astype(np.float32)
    
    if ENGINE.is_dual:
        inputs = {"sequence_input": seq_in, "stat_input": stat_in}
    else:
        inputs = seq_in
        
    out = ENGINE.model.predict(inputs, verbose=0, batch_size=256)
    return out.flatten().astype(np.float32)

def predict_one(readings: np.ndarray, customer_id: str = "SINGLE_USER", threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Single consumer prediction wrapper."""
    r = np.asarray(readings, dtype=np.float32).flatten().reshape(1, -1)
    probs = predict_sequences(r, threshold)
    prob = float(probs[0])
    res = classify(prob, threshold)
    res["customer_id"] = customer_id
    res["readings"] = r.flatten().tolist()
    res["threshold"] = threshold
    res["model_name"] = ENGINE.name
    return res
