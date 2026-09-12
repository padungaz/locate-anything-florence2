# ==============================================================================
# Multi-stage Dockerfile for locate-anything.cpp REST Service
# Stage 1: Build C++ Shared Library (liblocate_anything.so) with ggml static inside
# Stage 2: Minimal Python Runtime with FastAPI & Uvicorn
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source tree
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY third_party/ third_party/

# Compile shared library with -DLA_SHARED=ON
RUN cmake -B build_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DLA_SHARED=ON \
    -DLA_BUILD_CLI=OFF \
    -DLA_BUILD_TESTS=OFF \
    && cmake --build build_shared --config Release -j$(nproc)

# ------------------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------------------
FROM python:3.10-slim-bullseye AS runtime

LABEL maintainer="LocateAnything Development Team"
LABEL description="Production REST Service for LocateAnything-3B Visual Grounding C++ Engine"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH \
    LA_LIB_PATH=/app/lib/liblocate_anything.so \
    LA_MODEL_PATH=/app/models/locate-anything-q4_k.gguf \
    LA_THREADS=8 \
    LA_MODE=fast \
    LA_PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.31.0 \
    python-multipart==0.0.12 \
    Pillow==10.4.0 \
    requests==2.32.3

WORKDIR /app

# Copy compiled shared library from builder
COPY --from=builder /build/build_shared/liblocate_anything.so /app/lib/liblocate_anything.so

# Copy integration code & scripts
COPY integration/ /app/integration/
COPY docker/server_entrypoint.sh /app/docker/server_entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/docker/server_entrypoint.sh

# Create models mount point
RUN mkdir -p /app/models

EXPOSE 8080

# Health check using the readiness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ready || exit 1

ENTRYPOINT ["/app/docker/server_entrypoint.sh"]
