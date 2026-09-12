#!/bin/bash
set -e

echo "================================================================="
echo "  LocateAnything.cpp Vision Grounding REST Service (Production)  "
echo "================================================================="
echo "Model Path    : ${LA_MODEL_PATH:-/app/models/locate-anything-q4_k.gguf}"
echo "Library Path  : ${LA_LIB_PATH:-/app/lib/liblocate_anything.so}"
echo "CPU Threads   : ${LA_THREADS:-8}"
echo "Decode Mode   : ${LA_MODE:-fast}"
echo "Port          : ${LA_PORT:-8080}"
echo "================================================================="

# Check if model exists
MODEL_FILE="${LA_MODEL_PATH:-/app/models/locate-anything-q4_k.gguf}"
if [ ! -f "$MODEL_FILE" ]; then
    echo "[WARNING] Model file not found at: $MODEL_FILE"
    echo "[WARNING] Please mount your models directory to /app/models, e.g.:"
    echo "          docker run -v \$(pwd)/models:/app/models -p 8080:8080 locate-anything:latest"
fi

# Execute Uvicorn server (single worker to manage Singleton Engine safely)
exec python3 -m uvicorn integration.api_server:app \
    --host 0.0.0.0 \
    --port "${LA_PORT:-8080}" \
    --workers 1 \
    --log-level info
