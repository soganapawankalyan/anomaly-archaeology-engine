import numpy as np
import pandas as pd

np.random.seed(42)

SIGNALS = ["temperature", "pressure", "vibration", "error_rate", "throughput"]

SIGNAL_UNITS = {
    "temperature": "°C",
    "pressure":    "PSI",
    "vibration":   "mm/s",
    "error_rate":  "%",
    "throughput":  "units/hr",
}

SIGNAL_BASELINES = {
    "temperature": 72.0,
    "pressure":    14.5,
    "vibration":   2.1,
    "error_rate":  1.2,
    "throughput":  850.0,
}

SIGNAL_NOISE = {
    "temperature": 1.2,
    "pressure":    0.3,
    "vibration":   0.2,
    "error_rate":  0.15,
    "throughput":  18.0,
}


def _base(n: int) -> dict:
    return {s: SIGNAL_BASELINES[s] + np.random.normal(0, SIGNAL_NOISE[s], n)
            for s in SIGNALS}


def scenario_cooling_failure() -> dict:
    """
    Cooling system failure.
    Temperature rises first (t=80), pressure follows (t=85),
    vibration spikes as thermal stress builds (t=90),
    error_rate climbs (t=92), throughput drops (t=95).
    """
    n = 200
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
    d = _base(n)

    d["temperature"][80:]  += np.linspace(0, 28, n-80)
    d["pressure"][85:]     += np.linspace(0, 4.2, n-85)
    d["vibration"][90:]    += np.linspace(0, 6.8, n-90) + np.random.normal(0, 0.4, n-90)
    d["error_rate"][92:]   += np.linspace(0, 12.5, n-92)
    d["throughput"][95:]   -= np.linspace(0, 320, n-95)

    return {
        "name": "Cooling system failure",
        "description": "Progressive thermal overload — cooling unit degraded, triggering cascading stress across mechanical and operational systems.",
        "anomaly_idx": 80,
        "root_cause_signal": "temperature",
        "incident_type": "cascade",
        "dates": dates,
        "data": pd.DataFrame(d, index=dates),
        "severity": "critical",
    }


def scenario_pressure_spike() -> dict:
    """
    Sudden pressure surge from upstream supply.
    Pressure spikes first (t=60), vibration immediately follows (t=62),
    temperature rises from mechanical stress (t=70),
    error_rate jumps (t=72), throughput fluctuates (t=75).
    """
    n = 180
    dates = pd.date_range("2024-02-01 00:00", periods=n, freq="15min")
    d = _base(n)

    spike = np.zeros(n)
    spike[60:65] = np.array([8.2, 11.4, 9.8, 7.1, 5.3])
    d["pressure"]   += spike
    d["pressure"][65:] += np.linspace(2.1, 0.3, n-65)

    d["vibration"][62:] += spike[60:60+(n-62)][:n-62] * 0.4 + np.random.normal(0, 0.1, n-62)
    d["temperature"][70:] += np.linspace(0, 8.4, n-70)
    d["error_rate"][72:]  += np.linspace(0, 5.6, n-72)
    d["throughput"][75:]  += np.random.normal(0, 45, n-75)

    return {
        "name": "Upstream pressure surge",
        "description": "Sudden supply pressure spike from upstream valve malfunction — mechanical stress propagated through system within minutes.",
        "anomaly_idx": 60,
        "root_cause_signal": "pressure",
        "incident_type": "spike",
        "dates": dates,
        "data": pd.DataFrame(d, index=dates),
        "severity": "high",
    }


def scenario_bearing_wear() -> dict:
    """
    Slow bearing degradation — gradual drift, hardest to detect.
    Vibration drifts upward first (t=40), temperature follows slowly (t=60),
    error_rate gradually rises (t=80), throughput slowly declines (t=90).
    Pressure mostly unaffected.
    """
    n = 200
    dates = pd.date_range("2024-03-01 00:00", periods=n, freq="30min")
    d = _base(n)

    d["vibration"][40:]    += np.linspace(0, 9.2, n-40) + np.random.normal(0, 0.3, n-40)
    d["temperature"][60:]  += np.linspace(0, 14.5, n-60)
    d["error_rate"][80:]   += np.linspace(0, 8.8, n-80)
    d["throughput"][90:]   -= np.linspace(0, 180, n-90) + np.random.normal(0, 12, n-90)

    return {
        "name": "Bearing wear degradation",
        "description": "Slow mechanical degradation from bearing wear — gradual signal drift over hours before operational impact became visible.",
        "anomaly_idx": 40,
        "root_cause_signal": "vibration",
        "incident_type": "drift",
        "dates": dates,
        "data": pd.DataFrame(d, index=dates),
        "severity": "medium",
    }


def scenario_software_fault() -> dict:
    """
    Software/control system fault causing erratic behavior.
    Error rate spikes suddenly (t=70), throughput drops immediately (t=71),
    pressure becomes erratic (t=74), temperature slightly elevated (t=80).
    Vibration normal throughout.
    """
    n = 180
    dates = pd.date_range("2024-04-01 00:00", periods=n, freq="10min")
    d = _base(n)

    d["error_rate"][70:]   += np.random.exponential(4.5, n-70) + 6.2
    d["throughput"][71:]   -= np.linspace(0, 280, n-71) + np.random.normal(0, 25, n-71)
    d["pressure"][74:]     += np.random.normal(0, 1.8, n-74)
    d["temperature"][80:]  += np.linspace(0, 5.2, n-80)

    return {
        "name": "Control system fault",
        "description": "Software fault in control system triggered erratic process behavior — error cascade preceded mechanical response by several minutes.",
        "anomaly_idx": 70,
        "root_cause_signal": "error_rate",
        "incident_type": "fault",
        "dates": dates,
        "data": pd.DataFrame(d, index=dates),
        "severity": "high",
    }


ALL_SCENARIOS = {
    "Cooling system failure":   scenario_cooling_failure,
    "Upstream pressure surge":  scenario_pressure_spike,
    "Bearing wear degradation": scenario_bearing_wear,
    "Control system fault":     scenario_software_fault,
}


def get_scenario(name: str) -> dict:
    return ALL_SCENARIOS[name]()


if __name__ == "__main__":
    for name, fn in ALL_SCENARIOS.items():
        s = fn()
        df = s["data"]
        print(f"\n{s['name']}  [{s['severity'].upper()}]")
        print(f"  Type: {s['incident_type']}  |  Root cause: {s['root_cause_signal']}")
        print(f"  Rows: {len(df)}  |  Anomaly at index: {s['anomaly_idx']}")
        for sig in SIGNALS:
            print(f"  {sig:12s}: min={df[sig].min():.2f}  max={df[sig].max():.2f}  mean={df[sig].mean():.2f}")
