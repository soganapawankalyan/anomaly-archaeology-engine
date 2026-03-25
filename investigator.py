import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")


def extract_anomaly_window(df: pd.DataFrame, signals: list,
                           anomaly_idx: int, pre: int = 20,
                           post: int = 30) -> pd.DataFrame:
    start = max(0, anomaly_idx - pre)
    end   = min(len(df), anomaly_idx + post)
    return df.iloc[start:end][signals].copy()


def compute_cross_correlations(window_df: pd.DataFrame,
                                signals: list) -> pd.DataFrame:
    corr_matrix = pd.DataFrame(index=signals, columns=signals, dtype=float)
    for s1 in signals:
        for s2 in signals:
            if s1 == s2:
                corr_matrix.loc[s1, s2] = 1.0
            else:
                x = window_df[s1].values.astype(float)
                y = window_df[s2].values.astype(float)
                if np.std(x) == 0 or np.std(y) == 0:
                    corr_matrix.loc[s1, s2] = 0.0
                else:
                    r, _ = stats.pearsonr(x, y)
                    corr_matrix.loc[s1, s2] = round(r, 4)
    return corr_matrix


def compute_lag_profile(detection_results: dict,
                         signals: list) -> list:
    entries = []
    for sig in signals:
        first = detection_results[sig]["first_anomaly"]
        if first is not None:
            entries.append({
                "signal":       sig,
                "first_anomaly": first,
                "severity":     detection_results[sig]["severity_score"],
                "count":        detection_results[sig]["anomaly_count"],
            })
    entries.sort(key=lambda x: x["first_anomaly"])
    for i, e in enumerate(entries):
        if i == 0:
            e["lag_from_first"] = 0
        else:
            e["lag_from_first"] = e["first_anomaly"] - entries[0]["first_anomaly"]
    return entries


def run_granger_tests(window_df: pd.DataFrame, signals: list,
                       max_lag: int = 4) -> dict:
    granger_scores = {}
    for target in signals:
        scores = {}
        for cause in signals:
            if cause == target:
                continue
            try:
                test_data = window_df[[target, cause]].dropna()
                if len(test_data) < max_lag * 3 + 2:
                    scores[cause] = 0.0
                    continue
                result = grangercausalitytests(
                    test_data.values, maxlag=max_lag, verbose=False
                )
                min_pval = min(
                    result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                )
                scores[cause] = round(1 - min_pval, 4)
            except Exception:
                scores[cause] = 0.0
        granger_scores[target] = scores
    return granger_scores


def rank_root_causes(lag_profile: list, granger_scores: dict,
                      detection_results: dict) -> list:
    if not lag_profile:
        return []
    first_signal = lag_profile[0]["signal"]
    candidates = []
    for entry in lag_profile:
        sig = entry["signal"]
        lag_score = 1.0 / (1.0 + entry["lag_from_first"] * 0.1)
        granger_score = 0.0
        for target in granger_scores:
            if target != sig and sig in granger_scores[target]:
                granger_score = max(granger_score,
                                    granger_scores[target].get(sig, 0.0))
        severity_score = detection_results[sig]["severity_score"]
        composite = (
            lag_score       * 0.40 +
            granger_score   * 0.35 +
            severity_score  * 0.25
        )
        candidates.append({
            "signal":         sig,
            "composite_score": round(composite, 4),
            "lag_score":       round(lag_score, 4),
            "granger_score":   round(granger_score, 4),
            "severity_score":  round(severity_score, 4),
            "first_anomaly":   entry["first_anomaly"],
            "lag_from_first":  entry["lag_from_first"],
        })
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return candidates


def build_evidence_timeline(df: pd.DataFrame, detection_results: dict,
                             signals: list, scenario: dict) -> list:
    events = []
    for sig in signals:
        res = detection_results[sig]
        if res["first_anomaly"] is not None:
            idx = res["first_anomaly"]
            ts  = df.index[idx] if hasattr(df.index, '__getitem__') else idx
            atype = res["anomaly_types"].get(idx, "anomaly")
            events.append({
                "timestamp":     str(ts),
                "index":         idx,
                "signal":        sig,
                "event":         f"{atype.replace('_',' ').title()} detected",
                "severity_score": res["severity_score"],
                "value":         round(float(df[sig].iloc[idx]), 3),
            })
    events.sort(key=lambda x: x["index"])
    return events


def run_full_investigation(scenario: dict, detection_results: dict,
                            signals: list) -> dict:
    df          = scenario["data"]
    anomaly_idx = scenario["anomaly_idx"]
    window_df   = extract_anomaly_window(df, signals, anomaly_idx)
    corr_matrix = compute_cross_correlations(window_df, signals)
    lag_profile = compute_lag_profile(detection_results, signals)
    granger     = run_granger_tests(window_df, signals)
    root_causes = rank_root_causes(lag_profile, granger, detection_results)
    timeline    = build_evidence_timeline(df, detection_results,
                                          signals, scenario)
    return {
        "scenario":       scenario,
        "window_df":      window_df,
        "corr_matrix":    corr_matrix,
        "lag_profile":    lag_profile,
        "granger_scores": granger,
        "root_causes":    root_causes,
        "timeline":       timeline,
        "top_cause":      root_causes[0]["signal"] if root_causes else None,
    }


if __name__ == "__main__":
    from signals import ALL_SCENARIOS, SIGNALS
    from detector import detect_all_signals

    for name, fn in ALL_SCENARIOS.items():
        s = fn()
        det = detect_all_signals(s["data"], SIGNALS)
        inv = run_full_investigation(s, det, SIGNALS)

        print(f"\n{'='*55}")
        print(f"{s['name']}  [{s['severity'].upper()}]")
        print(f"True root cause:    {s['root_cause_signal']}")
        print(f"Detected top cause: {inv['top_cause']}")
        print(f"\nRoot cause ranking:")
        for rc in inv["root_causes"][:3]:
            marker = " <-- TOP" if rc == inv["root_causes"][0] else ""
            print(f"  {rc['signal']:12s}  composite={rc['composite_score']:.3f}"
                  f"  lag={rc['lag_score']:.3f}"
                  f"  granger={rc['granger_score']:.3f}"
                  f"  severity={rc['severity_score']:.3f}{marker}")
        print(f"\nEvidence timeline (first 5 events):")
        for ev in inv["timeline"][:5]:
            print(f"  [{ev['index']:3d}] {ev['signal']:12s} — {ev['event']}"
                  f"  (val={ev['value']})")
