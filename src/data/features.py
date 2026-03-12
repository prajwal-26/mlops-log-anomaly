

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from loguru import logger


# ─── Feature Config ───────────────────────────────────────
# One place to control everything about features

@dataclass
class FeatureConfig:
    window_seconds:   int   = 60      # 1 minute windows
    min_logs_window:  int   = 5       # skip windows with < 5 logs
    repeated_error_threshold: int = 3 # same msg N times = repeated

    # These are the EXACT features fed to the model
    # Order matters — must be identical at train and serve time
    feature_columns: list = None

    def __post_init__(self):
        self.feature_columns = [
            "request_rate",
            "error_count",
            "warn_count",
            "unique_endpoints",
            "latency_mean",
            "latency_p99",
            "latency_max",
            "latency_std",
            "error_rate",
            "status_5xx_rate",
            "status_4xx_rate",
            "unique_users",
            "unique_ips",
            "repeated_errors",
        ]


# ─── Core Extraction Function ─────────────────────────────

def extract_features(window: list[dict],
                     config: FeatureConfig = None) -> dict | None:
    """
    Extract numeric features from a list of log line dicts.

    This is the SINGLE function called at both:
      - training time (batch, from parquet)
      - serving time  (real time, from API request)

    Args:
        window: list of log line dicts from a time window
        config: FeatureConfig (uses defaults if None)

    Returns:
        dict of feature name → float value
        None if window is too small to be meaningful
    """
    cfg = config or FeatureConfig()

    if len(window) < cfg.min_logs_window:
        return None

    total     = len(window)
    latencies = [l["latency_ms"] for l in window]
    levels    = [l["level"] for l in window]
    statuses  = [str(l["status"]) for l in window]
    paths     = [l["path"] for l in window]
    users     = [l["user_id"] for l in window]
    ips       = [l["ip"] for l in window]
    errors    = [l for l in window if l["level"] == "ERROR"]
    warns     = [l for l in window if l["level"] == "WARN"]

    # Repeated error messages
    error_msgs     = [l.get("error") for l in errors
                      if l.get("error")]
    msg_counts     = Counter(error_msgs)
    repeated_errors = sum(
        1 for count in msg_counts.values()
        if count >= cfg.repeated_error_threshold
    )

    features = {
        # ── Volume ────────────────────────────────────────
        "request_rate":    total / cfg.window_seconds,
        "error_count":     float(len(errors)),
        "warn_count":      float(len(warns)),
        "unique_endpoints": float(len(set(paths))),

        # ── Latency ───────────────────────────────────────
        "latency_mean":    float(np.mean(latencies)),
        "latency_p99":     float(np.percentile(latencies, 99)),
        "latency_max":     float(np.max(latencies)),
        "latency_std":     float(np.std(latencies)),

        # ── Error Rates ───────────────────────────────────
        "error_rate":      len(errors) / total,
        "status_5xx_rate": sum(
                               1 for s in statuses
                               if s.startswith("5")
                           ) / total,
        "status_4xx_rate": sum(
                               1 for s in statuses
                               if s.startswith("4")
                           ) / total,

        # ── Patterns ──────────────────────────────────────
        "unique_users":    float(len(set(users))),
        "unique_ips":      float(len(set(ips))),
        "repeated_errors": float(repeated_errors),
    }

    return features


def extract_features_from_df(df: pd.DataFrame,
                              config: FeatureConfig = None
                              ) -> pd.DataFrame:
    """
    Batch extraction from a DataFrame of raw logs.
    Groups by window_start and extracts features per window.
    Used during training data preparation.
    """
    cfg    = config or FeatureConfig()
    result = []

    for window_start, group in df.groupby("window_start"):
        logs = group.to_dict("records")
        feat = extract_features(logs, cfg)

        if feat is not None:
            feat["window_start"] = window_start
            # Keep label for evaluation
            feat["is_anomaly"] = group["is_anomaly"].any() \
                                  if "is_anomaly" in group.columns \
                                  else None
            result.append(feat)

    return pd.DataFrame(result)


def get_feature_vector(features: dict,
                       config: FeatureConfig = None
                       ) -> np.ndarray:
    """
    Convert feature dict → numpy array for model input.

    CRITICAL: Column order must be identical at train and serve time.
    This function guarantees that by using config.feature_columns.
    """
    cfg = config or FeatureConfig()

    vector = np.array([
        features[col] for col in cfg.feature_columns
    ], dtype=np.float32)

    return vector.reshape(1, -1)   # shape (1, n_features)


def get_feature_names(config: FeatureConfig = None) -> list[str]:
    """
    Returns the ordered list of feature names.
    Used by MLflow to log feature schema.
    """
    cfg = config or FeatureConfig()
    return cfg.feature_columns


# ─── Sanity Check ─────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test — verify extraction works end to end

    from datetime import datetime, timedelta
    import random

    logger.info("Running feature extraction smoke test...")

    # Build a fake window of 50 log lines
    now        = datetime.now()
    fake_window = []

    for i in range(50):
        ts = now + timedelta(seconds=random.uniform(0, 59))
        fake_window.append({
            "timestamp":  ts.isoformat(),
            "level":      random.choice(
                              ["INFO"] * 8 + ["ERROR"] * 2
                          ),
            "path":       random.choice(["/predict", "/health"]),
            "status":     random.choice([200] * 8 + [500] * 2),
            "latency_ms": int(np.random.lognormal(3.5, 0.4)),
            "user_id":    f"u{random.randint(1, 100):04d}",
            "ip":         f"10.0.{random.randint(1,5)}.1",
            "error":      None,
        })

    features = extract_features(fake_window)
    vector   = get_feature_vector(features)

    logger.info("Extracted features:")
    for name, val in features.items():
        logger.info(f"  {name:<22} = {val:.4f}")

    logger.info(f"Feature vector shape: {vector.shape}")
    logger.info(f"Feature names: {get_feature_names()}")
    logger.success("Smoke test passed ✅")