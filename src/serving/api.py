
import os
import sys
import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from loguru import logger
from dotenv import load_dotenv

# Always run from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter, Histogram, Gauge,
    make_asgi_app, multiprocess,
    CollectorRegistry
)

import mlflow
import mlflow.sklearn

load_dotenv()


# ─── Config ───────────────────────────────────────────────

MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI",
                          "http://127.0.0.1:5000")
MODEL_NAME   = os.getenv("MODEL_NAME",
                          "log-anomaly-detector")
MODEL_ALIAS  = os.getenv("MODEL_ALIAS", "production")


# ─── Prometheus Metrics ───────────────────────────────────
# These are what Grafana will visualise

REQUEST_COUNT = Counter(
    "anomaly_requests_total",
    "Total scoring requests",
    ["status"]          # label: success or error
)

LATENCY = Histogram(
    "anomaly_scoring_latency_seconds",
    "Time to score one request",
    buckets=[0.001, 0.005, 0.01,
             0.025, 0.05, 0.1, 0.25, 0.5]
)

ANOMALY_COUNTER = Counter(
    "anomalies_detected_total",
    "Total anomalies flagged",
    ["severity"]        # label: HIGH, MEDIUM, LOW
)

ANOMALY_SCORE = Histogram(
    "anomaly_score_distribution",
    "Distribution of raw anomaly scores",
    buckets=[-0.5, -0.4, -0.3, -0.2,
             -0.1, 0.0, 0.1, 0.2, 0.3]
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if model is loaded, 0 if not"
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently loaded model version",
    ["version", "stage"]
)


# ─── Global Model State ───────────────────────────────────

model         = None
model_version = None


# ─── Lifespan ─────────────────────────────────────────────
# Runs on startup and shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    await load_model()
    yield
    # SHUTDOWN
    logger.info("Shutting down API")


async def load_model():
    """Load model from MLflow registry at startup."""
    global model, model_version

    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"Loading model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        MODEL_LOADED.set(1)

        # Get version info for metrics
        client  = mlflow.tracking.MlflowClient()
        version_info  = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
            )
        model_version = version_info.version
        MODEL_VERSION.labels(
            version = str(model_version),
            stage   = MODEL_ALIAS
            ).set(1)

        logger.success(
            f"Model loaded: {MODEL_NAME} "
            f"v{model_version} ({MODEL_ALIAS})"
        )

    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Failed to load model: {e}")
        logger.warning("API starting without model — "
                       "predictions will fail until model "
                       "is available")


# ─── App ──────────────────────────────────────────────────

app = FastAPI(
    title       = "Log Anomaly Detection API",
    description = "Scores log windows for anomalies",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── Request / Response Schema ────────────────────────────

class LogFeatures(BaseModel):
    """
    Feature vector extracted from a 60-second log window.
    Pydantic validates ranges automatically.
    Invalid input → HTTP 422, model never called.
    """
    request_rate:    float = Field(gt=0,   lt=1000,
                         description="Requests per second")
    error_count:     float = Field(ge=0,
                         description="Error log count")
    warn_count:      float = Field(ge=0,
                         description="Warning log count")
    unique_endpoints:float = Field(ge=0,
                         description="Unique paths hit")
    latency_mean:    float = Field(gt=0,   lt=60000,
                         description="Mean latency ms")
    latency_p99:     float = Field(gt=0,   lt=60000,
                         description="P99 latency ms")
    latency_max:     float = Field(gt=0,   lt=60000,
                         description="Max latency ms")
    latency_std:     float = Field(ge=0,
                         description="Latency std dev")
    error_rate:      float = Field(ge=0,   le=1,
                         description="Error rate 0-1")
    status_5xx_rate: float = Field(ge=0,   le=1,
                         description="5xx rate 0-1")
    status_4xx_rate: float = Field(ge=0,   le=1,
                         description="4xx rate 0-1")
    unique_users:    float = Field(ge=0,
                         description="Unique user IDs")
    unique_ips:      float = Field(ge=0,
                         description="Unique IPs")
    repeated_errors: float = Field(ge=0,
                         description="Repeated error msgs")


class PredictionResponse(BaseModel):
    is_anomaly:    bool
    anomaly_score: float
    severity:      str    # NORMAL, LOW, MEDIUM, HIGH
    message:       str


# ─── Endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Liveness probe — is the API alive?
    Kubernetes calls this every 10 seconds.
    Returns 200 if alive, 503 if not.
    """
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """
    Readiness probe — is the API ready to serve?
    Kubernetes calls this before sending traffic.
    Returns 200 only if model is loaded.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    return {
        "status":        "ready",
        "model":         MODEL_NAME,
        "model_version": model_version,
        "stage":         MODEL_ALIAS,
    }


@app.post("/score", response_model=PredictionResponse)
async def score(features: LogFeatures):
    """
    Score a log window for anomalies.
    Input:  14 numeric features from a 60s log window
    Output: anomaly flag, score, severity
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # Build feature vector
        # Order must match training exactly
        X = np.array([[
            features.request_rate,
            features.error_count,
            features.warn_count,
            features.unique_endpoints,
            features.latency_mean,
            features.latency_p99,
            features.latency_max,
            features.latency_std,
            features.error_rate,
            features.status_5xx_rate,
            features.status_4xx_rate,
            features.unique_users,
            features.unique_ips,
            features.repeated_errors,
        ]], dtype=np.float32)

        # Score — decision_function returns:
        #   negative = more anomalous
        #   positive = more normal
        raw_score  = float(model.decision_function(X)[0])
        prediction = model.predict(X)[0]
        is_anomaly = prediction == -1

        # Severity based on score
        if not is_anomaly:
            severity = "NORMAL"
        elif raw_score < -0.3:
            severity = "HIGH"
        elif raw_score < -0.1:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Update Prometheus metrics
        if is_anomaly:
            ANOMALY_COUNTER.labels(
                severity=severity
            ).inc()

        ANOMALY_SCORE.observe(raw_score)
        REQUEST_COUNT.labels(status="success").inc()
        LATENCY.observe(time.time() - start_time)

        message = (
            f"Anomaly detected — severity {severity}"
            if is_anomaly
            else "Normal window"
        )

        return PredictionResponse(
            is_anomaly    = bool(is_anomaly),
            anomaly_score = round(raw_score, 4),
            severity      = severity,
            message       = message,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Scoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ─── Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
    )
