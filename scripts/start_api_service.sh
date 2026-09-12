#!/bin/bash
# ==============================================================================
# start_api_service.sh — Launch LocateAnything REST API Server Locally
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

export LD_LIBRARY_PATH="${ROOT_DIR}/build_shared:${LD_LIBRARY_PATH}"
export LA_LIB_PATH="${ROOT_DIR}/build_shared/liblocate_anything.so"
export LA_MODEL_PATH="${ROOT_DIR}/models/locate-anything-q4_k.gguf"
export LA_THREADS="${LA_THREADS:-8}"
export LA_MODE="${LA_MODE:-fast}"
export LA_PORT="${LA_PORT:-8080}"

echo "================================================================="
echo "  Starting LocateAnything REST API Service                       "
echo "================================================================="
echo "Model Path   : ${LA_MODEL_PATH}"
echo "Library Path : ${LA_LIB_PATH}"
echo "CPU Threads  : ${LA_THREADS}"
echo "Decode Mode  : ${LA_MODE}"
echo "Port         : ${LA_PORT}"
echo "================================================================="

if [ ! -f "${LA_LIB_PATH}" ]; then
    echo "[ERROR] Shared library not found: ${LA_LIB_PATH}"
    echo "Please build it first: cmake -B build_shared -DLA_SHARED=ON && cmake --build build_shared"
    exit 1
fi

if [ ! -f "${LA_MODEL_PATH}" ]; then
    echo "[ERROR] Model file not found: ${LA_MODEL_PATH}"
    exit 1
fi

exec python3 -m uvicorn integration.api_server:app \
    --host 0.0.0.0 \
    --port "${LA_PORT}" \
    --workers 1
