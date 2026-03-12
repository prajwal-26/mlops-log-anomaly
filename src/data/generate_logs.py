import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from loguru import logger


# ─── Constants ────────────────────────────────────────────

ENDPOINTS = [
    "/predict", "/health", "/metrics",
    "/api/v1/score", "/api/v1/batch"
]

NORMAL_USERS = [f"u{i:04d}" for i in range(1, 500)]
NORMAL_IPS   = [f"10.0.{i}.{j}" for i in range(1, 10)
                                 for j in range(1, 30)]

ERROR_MESSAGES = [
    "DB connection timeout",
    "upstream service unavailable",
    "memory allocation failed",
    "model inference timeout",
    "feature store unreachable",
]


# ─── Log Line Generators ──────────────────────────────────

def normal_log(timestamp: datetime) -> dict:
    """Generate one normal log line."""
    return {
        "timestamp": timestamp.isoformat(),
        "level":     "INFO",
        "service":   "api-server",
        "req_id":    f"req_{random.randint(100000, 999999)}",
        "method":    random.choice(["GET", "POST"]),
        "path":      random.choice(ENDPOINTS),
        "status":    random.choice([200, 200, 200, 201, 204]),
        "latency_ms": int(np.random.lognormal(mean=3.5, sigma=0.4)),
        "user_id":   random.choice(NORMAL_USERS),
        "ip":        random.choice(NORMAL_IPS),
        "error":     None,
    }


def anomaly_spike_log(timestamp: datetime) -> dict:
    """Generate one anomalous log — latency spike + errors."""
    return {
        "timestamp": timestamp.isoformat(),
        "level":     "ERROR",
        "service":   "api-server",
        "req_id":    f"req_{random.randint(100000, 999999)}",
        "method":    "POST",
        "path":      "/predict",
        "status":    random.choice([500, 502, 503]),
        "latency_ms": int(np.random.lognormal(mean=8.5, sigma=0.3)),
        "user_id":   random.choice(NORMAL_USERS),
        "ip":        random.choice(NORMAL_IPS),
        "error":     random.choice(ERROR_MESSAGES),
    }


def anomaly_flood_log(timestamp: datetime) -> dict:
    """Generate one log from a traffic flood attack."""
    return {
        "timestamp": timestamp.isoformat(),
        "level":     "WARN",
        "service":   "api-server",
        "req_id":    f"req_{random.randint(100000, 999999)}",
        "method":    "POST",
        "path":      "/predict",
        "status":    429,
        "latency_ms": random.randint(5, 25),
        "user_id":   f"u_attacker_{random.randint(1,5)}",
        "ip":        f"192.168.{random.randint(1,5)}.1",
        "error":     "rate limit exceeded",
    }


# ─── Window Feature Extractor ─────────────────────────────

def extract_window_features(window: list[dict]) -> dict:
    """
    Takes a list of log lines (1 minute window).
    Returns a flat feature vector as a dict.
    This is what the model actually trains on.
    """
    if not window:
        return None

    total      = len(window)
    errors     = [l for l in window if l["level"] == "ERROR"]
    warns      = [l for l in window if l["level"] == "WARN"]
    latencies  = [l["latency_ms"] for l in window]
    status_5xx = [l for l in window if str(l["status"]).startswith("5")]
    status_4xx = [l for l in window if str(l["status"]).startswith("4")]

    # Count repeated error messages
    error_msgs = [l["error"] for l in errors if l["error"]]
    from collections import Counter
    msg_counts     = Counter(error_msgs)
    repeated_errors = sum(1 for c in msg_counts.values() if c >= 3)

    return {
        # Volume
        "request_rate":       total / 60.0,
        "error_count":        len(errors),
        "warn_count":         len(warns),
        "unique_endpoints":   len(set(l["path"] for l in window)),

        # Latency
        "latency_mean":       float(np.mean(latencies)),
        "latency_p99":        float(np.percentile(latencies, 99)),
        "latency_max":        float(np.max(latencies)),
        "latency_std":        float(np.std(latencies)),

        # Error rates
        "error_rate":         len(errors) / total,
        "status_5xx_rate":    len(status_5xx) / total,
        "status_4xx_rate":    len(status_4xx) / total,

        # Pattern
        "unique_users":       len(set(l["user_id"] for l in window)),
        "unique_ips":         len(set(l["ip"] for l in window)),
        "repeated_errors":    repeated_errors,

        # Label — for evaluation only, NOT fed to model
        "is_anomaly":         any(l["level"] == "ERROR"
                                  and l["latency_ms"] > 1000
                                  for l in window),
    }


# ─── Dataset Generator ────────────────────────────────────

def generate_dataset(days: int = 30,
                     anomaly_rate: float = 0.05,
                     output_dir: str = "data/raw") -> None:
    """
    Generate `days` days of synthetic logs.
    Injects anomalies at roughly `anomaly_rate` of windows.
    Saves raw JSONL + windowed features as parquet.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {days} days of logs...")

    raw_logs    = []
    start_time  = datetime.now() - timedelta(days=days)
    current     = start_time
    window_size = timedelta(seconds=60)
    features    = []

    total_windows = days * 24 * 60   # one window per minute

    for i in range(total_windows):
        window_start = current
        window_end   = current + window_size

        # Decide: normal window or anomaly window?
        is_anomaly_window = random.random() < anomaly_rate

        # Generate 10–150 log lines per window
        n_logs = random.randint(10, 150)
        window_logs = []

        for _ in range(n_logs):
            ts = window_start + timedelta(
                seconds=random.uniform(0, 59)
            )
            if is_anomaly_window:
                # Mix of anomaly types
                r = random.random()
                if r < 0.6:
                    log = anomaly_spike_log(ts)
                else:
                    log = anomaly_flood_log(ts)
            else:
                log = normal_log(ts)

            window_logs.append(log)
            raw_logs.append(log)

        # Extract features from this window
        feat = extract_window_features(window_logs)
        if feat:
            feat["window_start"] = window_start.isoformat()
            features.append(feat)

        current = window_end

        if i % 1000 == 0:
            logger.info(f"  Progress: {i}/{total_windows} windows")

    # Save raw logs as JSONL
    raw_path = Path(output_dir) / "logs.jsonl"
    with open(raw_path, "w") as f:
        for log in raw_logs:
            f.write(json.dumps(log) + "\n")
    logger.info(f"Raw logs saved → {raw_path} ({len(raw_logs):,} lines)")

    # Save features as parquet
    df = pd.DataFrame(features)

    # Train / test split — last 7 days = test
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    train_path = processed_dir / "train.parquet"
    test_path  = processed_dir / "test.parquet"
    ref_path   = Path("data/reference") / "reference.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,   index=False)

    # Reference = first 7 days of training data
    # Used later for drift detection
    train_df.iloc[:10080].to_parquet(ref_path, index=False)

    logger.info(f"Train features → {train_path} ({len(train_df):,} rows)")
    logger.info(f"Test features  → {test_path}  ({len(test_df):,} rows)")
    logger.info(f"Reference data → {ref_path}   (drift baseline)")

    # Quick sanity check
    anomaly_pct = df["is_anomaly"].mean() * 100
    logger.info(f"Anomaly rate in dataset: {anomaly_pct:.1f}%")


# ─── Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",         type=int,   default=30)
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--output-dir",   type=str,   default="data/raw")
    args = parser.parse_args()

    generate_dataset(
        days=args.days,
        anomaly_rate=args.anomaly_rate,
        output_dir=args.output_dir,
    )