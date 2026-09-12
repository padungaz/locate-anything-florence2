#!/bin/bash
# ==============================================================================
# run_cluster_daemon.sh — Run Florence-2 Cluster in Foreground / Daemon
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="/tmp/florence_cluster"
mkdir -p "${LOG_DIR}"

fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
sleep 1

MODEL_ID="${FLORENCE_MODEL_ID:-microsoft/Florence-2-large}"

cleanup() {
    echo "[DAEMON] Shutting down..."
    kill ${PID_W1} ${PID_LB} 2>/dev/null || true
    fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "[DAEMON] Starting Florence-2 Worker on :8081..."
env PYTHONPATH="${ROOT_DIR}" FLORENCE_MODEL_ID="${MODEL_ID}" FLORENCE_PORT=8081 \
  python3 -m uvicorn integration.api_server:app --host 0.0.0.0 --port 8081 --workers 1 > "${LOG_DIR}/worker_8081.log" 2>&1 &
PID_W1=$!

echo "[DAEMON] Starting Load Balancer on :8080..."
env PYTHONPATH="${ROOT_DIR}" CLUSTER_WORKERS="Florence2-GPU:http://127.0.0.1:8081:gpu" \
  python3 -m uvicorn integration.load_balancer:app --host 0.0.0.0 --port 8080 --workers 1 > "${LOG_DIR}/load_balancer.log" 2>&1 &
PID_LB=$!

echo "[DAEMON] Cluster running: Worker (${PID_W1}), Load Balancer (${PID_LB})"
wait ${PID_LB}
