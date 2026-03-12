
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass


# ─── Validation Config ────────────────────────────────────
# All thresholds in one place — easy to tune

@dataclass
class ValidationConfig:
    # Schema
    required_columns: list = None
    
    # Size
    min_rows: int = 5000

    # Anomaly rate — if outside this range data is suspicious
    min_anomaly_rate: float = 0.01   # at least 1% anomalies
    max_anomaly_rate: float = 0.30   # not more than 30%

    # Latency — physically impossible values = bad data
    max_latency_mean:  float = 5000.0   # ms
    min_latency_mean:  float = 1.0      # ms
    max_latency_p99:   float = 30000.0  # ms

    # Null tolerance
    max_null_rate: float = 0.01   # max 1% nulls per column

    # Request rate sanity
    min_request_rate: float = 0.1   # at least 1 req per 10 sec
    max_request_rate: float = 1000.0 # not more than 1000/sec

    def __post_init__(self):
        self.required_columns = [
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
            "is_anomaly",
            "window_start",
        ]


# ─── Individual Checks ────────────────────────────────────

class DataValidator:

    def __init__(self, config: ValidationConfig = None):
        self.config  = config or ValidationConfig()
        self.errors  = []
        self.warnings = []

    def _fail(self, message: str):
        """Record a hard failure — pipeline must stop."""
        self.errors.append(message)
        logger.error(f"❌ FAIL: {message}")

    def _warn(self, message: str):
        """Record a warning — pipeline continues but team is alerted."""
        self.warnings.append(message)
        logger.warning(f"⚠️  WARN: {message}")

    def _pass(self, message: str):
        logger.success(f"✅ PASS: {message}")

    # ── Check 1: Schema ───────────────────────────────────
    def check_schema(self, df: pd.DataFrame):
        missing = [c for c in self.config.required_columns
                   if c not in df.columns]
        if missing:
            self._fail(f"Missing columns: {missing}")
        else:
            self._pass(f"All {len(self.config.required_columns)}"
                       f" required columns present")

        # Check types make sense
        numeric_cols = [
            "request_rate", "latency_mean",
            "latency_p99", "error_rate"
        ]
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self._fail(f"Column {col} should be numeric,"
                               f" got {df[col].dtype}")

    # ── Check 2: Size ─────────────────────────────────────
    def check_size(self, df: pd.DataFrame):
        if len(df) < self.config.min_rows:
            self._fail(f"Dataset too small: {len(df):,} rows"
                       f" (min: {self.config.min_rows:,})")
        else:
            self._pass(f"Dataset size OK: {len(df):,} rows")

    # ── Check 3: Nulls ────────────────────────────────────
    def check_nulls(self, df: pd.DataFrame):
        for col in self.config.required_columns:
            if col not in df.columns:
                continue
            null_rate = df[col].isnull().mean()
            if null_rate > self.config.max_null_rate:
                self._fail(f"Column '{col}' has {null_rate:.1%}"
                           f" nulls (max: {self.config.max_null_rate:.1%})")
        
        total_null_rate = df.isnull().mean().mean()
        if total_null_rate == 0:
            self._pass("No null values found")

    # ── Check 4: Anomaly Rate ─────────────────────────────
    def check_anomaly_rate(self, df: pd.DataFrame):
        if "is_anomaly" not in df.columns:
            return

        rate = df["is_anomaly"].mean()

        if rate < self.config.min_anomaly_rate:
            self._fail(f"Anomaly rate too low: {rate:.2%}"
                       f" (min: {self.config.min_anomaly_rate:.2%})"
                       f" — data may be mislabelled or filtered")

        elif rate > self.config.max_anomaly_rate:
            self._fail(f"Anomaly rate too high: {rate:.2%}"
                       f" (max: {self.config.max_anomaly_rate:.2%})"
                       f" — training on mostly anomalies won't work"
                       f" for Isolation Forest")
        else:
            self._pass(f"Anomaly rate healthy: {rate:.2%}")

    # ── Check 5: Latency Sanity ───────────────────────────
    def check_latency(self, df: pd.DataFrame):
        if "latency_mean" not in df.columns:
            return

        mean_lat = df["latency_mean"].mean()
        p99_lat  = df["latency_p99"].max()
        neg_lat  = (df["latency_mean"] < 0).sum()

        if neg_lat > 0:
            self._fail(f"Found {neg_lat} rows with negative"
                       f" latency — impossible value")

        if mean_lat < self.config.min_latency_mean:
            self._warn(f"Mean latency very low: {mean_lat:.1f}ms"
                       f" — might be synthetic/mock data")

        if mean_lat > self.config.max_latency_mean:
            self._fail(f"Mean latency too high: {mean_lat:.1f}ms"
                       f" — data might be corrupted")

        if p99_lat > self.config.max_latency_p99:
            self._warn(f"P99 latency very high: {p99_lat:.1f}ms"
                       f" — check for outliers")

        if neg_lat == 0 and mean_lat <= self.config.max_latency_mean:
            self._pass(f"Latency values sane:"
                       f" mean={mean_lat:.1f}ms, p99={p99_lat:.1f}ms")

    # ── Check 6: Request Rate ─────────────────────────────
    def check_request_rate(self, df: pd.DataFrame):
        if "request_rate" not in df.columns:
            return

        min_rr = df["request_rate"].min()
        max_rr = df["request_rate"].max()

        if min_rr < self.config.min_request_rate:
            self._warn(f"Some windows have very low request rate:"
                       f" {min_rr:.2f} req/s")

        if max_rr > self.config.max_request_rate:
            self._warn(f"Some windows have very high request rate:"
                       f" {max_rr:.1f} req/s — possible flood attack"
                       f" or data error")

        self._pass(f"Request rate range:"
                   f" {min_rr:.1f} – {max_rr:.1f} req/s")

    # ── Check 7: Rates Are Valid Percentages ──────────────
    def check_rate_columns(self, df: pd.DataFrame):
        rate_cols = ["error_rate", "status_5xx_rate", "status_4xx_rate"]
        for col in rate_cols:
            if col not in df.columns:
                continue
            invalid = ((df[col] < 0) | (df[col] > 1)).sum()
            if invalid > 0:
                self._fail(f"Column '{col}' has {invalid} values"
                           f" outside [0, 1] — not a valid rate")
        
        self._pass("All rate columns within [0, 1]")


    # ── Run All Checks ────────────────────────────────────
    def validate(self, df: pd.DataFrame) -> bool:
        logger.info(f"Running validation on {len(df):,} rows...")
        logger.info("─" * 50)

        self.check_schema(df)
        self.check_size(df)
        self.check_nulls(df)
        self.check_anomaly_rate(df)
        self.check_latency(df)
        self.check_request_rate(df)
        self.check_rate_columns(df)

        logger.info("─" * 50)

        if self.warnings:
            logger.warning(f"{len(self.warnings)} warning(s) found")

        if self.errors:
            logger.error(f"{len(self.errors)} error(s) found"
                         f" — pipeline must stop")
            return False

        logger.success("All validation checks passed ✅")
        return True


# ─── Entry Point ──────────────────────────────────────────

def validate_data(data_path: str) -> bool:
    path = Path(data_path)

    if not path.exists():
        logger.error(f"File not found: {data_path}")
        return False

    logger.info(f"Loading: {data_path}")
    df = pd.read_parquet(path)

    validator = DataValidator()
    passed    = validator.validate(df)

    if not passed:
        sys.exit(1)   # Non-zero exit = Airflow marks task as FAILED

    return True


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 \
                else "data/processed/train.parquet"
    validate_data(data_path)
