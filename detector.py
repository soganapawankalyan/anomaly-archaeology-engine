import numpy as np
import pandas as pd
from scipy import stats


def detect_zscore(series: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    mean = np.mean(series)
    std  = np.std(series)
    if std == 0:
        return np.zeros(len(series), dtype=bool)
    z = np.abs((series - mean) / std)
    return z > threshold


def detect_iqr(series: np.ndarray, multiplier: float = 2.5) -> np.ndarray:
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_cusum(series: np.ndarray, threshold: float = 5.0,
                 drift: float = 0.5) -> np.ndarray:
    mean = np.mean(series[:max(10, len(series)//4)])
    std  = np.std(series[:max(10, len(series)//4)])
    if std == 0:
        return np.zeros(len(series), dtype=bool)
    normalized = (series - mean) / std
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    for i in range(1, len(series)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - drift)
        cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - drift)
    return (cusum_pos > threshold) | (cusum_neg > threshold)


def classify_anomaly_type(series: np.ndarray, idx: int,
                           window: int = 10) -> str:
    if idx < window or idx > len(series) - window:
        return "edge"
    pre  = series[max(0, idx-window):idx]
    post = series[idx:min(len(series), idx+window)]
    pre_mean  = np.mean(pre)
    post_mean = np.mean(post)
    pre_std   = np.std(pre)
    point_val = series[idx]
    if abs(point_val - pre_mean) > 3 * pre_std:
        return "spike"
    level_shift = abs(post_mean - pre_mean)
    if level_shift > 2 * pre_std:
        return "level_shift"
    slope = np.polyfit(range(len(post)), post, 1)[0]
    if abs(slope) > pre_std * 0.3:
        return "drift"
    if np.any(np.isnan(series[idx:idx+3])) or np.std(post) < pre_std * 0.1:
        return "dropout"
    return "anomaly"


def score_severity(series: np.ndarray, anomaly_mask: np.ndarray) -> float:
    if not np.any(anomaly_mask):
        return 0.0
    mean = np.mean(series)
    std  = np.std(series)
    if std == 0:
        return 0.0
    anomaly_vals = series[anomaly_mask]
    max_dev = np.max(np.abs(anomaly_vals - mean)) / std
    count_pct = np.sum(anomaly_mask) / len(series)
    score = min(1.0, (max_dev / 8.0) * 0.7 + count_pct * 0.3)
    return round(score, 4)


def detect_all_signals(df: pd.DataFrame,
                       signals: list) -> dict:
    results = {}
    for sig in signals:
        series = df[sig].values.astype(float)
        z_mask   = detect_zscore(series)
        iqr_mask = detect_iqr(series)
        cusum_mask = detect_cusum(series)
        combined = z_mask | iqr_mask | cusum_mask
        anomaly_indices = np.where(combined)[0].tolist()
        anomaly_types = {}
        for idx in anomaly_indices:
            anomaly_types[idx] = classify_anomaly_type(series, idx)
        severity_score = score_severity(series, combined)
        first_anomaly = int(anomaly_indices[0]) if anomaly_indices else None
        results[sig] = {
            "anomaly_mask":    combined,
            "z_mask":          z_mask,
            "iqr_mask":        iqr_mask,
            "cusum_mask":      cusum_mask,
            "anomaly_indices": anomaly_indices,
            "anomaly_types":   anomaly_types,
            "severity_score":  severity_score,
            "anomaly_count":   int(np.sum(combined)),
            "first_anomaly":   first_anomaly,
        }
    return results


def get_first_anomaly_signal(detection_results: dict) -> tuple:
    earliest = None
    earliest_sig = None
    for sig, res in detection_results.items():
        if res["first_anomaly"] is not None:
            if earliest is None or res["first_anomaly"] < earliest:
                earliest = res["first_anomaly"]
                earliest_sig = sig
    return earliest_sig, earliest


if __name__ == "__main__":
    from signals import ALL_SCENARIOS, SIGNALS
    for name, fn in ALL_SCENARIOS.items():
        s = fn()
        results = detect_all_signals(s["data"], SIGNALS)
        first_sig, first_idx = get_first_anomaly_signal(results)
        print(f"\n{s['name']}")
        print(f"  First anomaly: {first_sig} at index {first_idx} "
              f"(true root cause: {s['root_cause_signal']} at {s['anomaly_idx']})")
        for sig in SIGNALS:
            r = results[sig]
            if r["anomaly_count"] > 0:
                types = set(r["anomaly_types"].values())
                print(f"  {sig:12s}: {r['anomaly_count']:3d} anomalies | "
                      f"severity={r['severity_score']:.3f} | "
                      f"first={r['first_anomaly']:3d} | types={types}")
