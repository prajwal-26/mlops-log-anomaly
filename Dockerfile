# ── Stage 1: Builder ──────────────────────────────────────
# Install deps in separate stage so final image is smaller
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────
# Clean image with only what's needed to run
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

# Copy application code only
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]


# FROM python:3.12-slim

# WORKDIR /app

# # Install build deps, python packages, then clean up — all one layer
# RUN apt-get update && apt-get install -y \
#     gcc g++ \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirement.txt .
# RUN pip install --no-cache-dir -r requirement.txt

# COPY src/ ./src/

# EXPOSE 8000

# CMD ["uvicorn", "src.serving.api:app", \
#      "--host", "0.0.0.0", \
#      "--port", "8000"]