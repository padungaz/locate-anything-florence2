#!/bin/bash
# ==============================================================================
# start_cluster.sh — Launch Florence-2 Vision Grounding Cluster (Commercial)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
mkdir -p /tmp/florence_cluster

echo "================================================================="
echo "  STARTING FLORENCE-2 VISION GROUNDING CLUSTER (COMMERCIAL)      "
echo "================================================================="

# Stop any existing processes on ports 8080, 8081
echo "[INFO] Clearing ports 8080, 8081..."
fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
sleep 1

MODEL_ID="${FLORENCE_MODEL_ID:-microsoft/Florence-2-large}"

# 1. Start Worker 1 (GPU Primary on :8081)
echo "[INFO] Starting Florence-2 Worker on port 8081..."
nohup env \
  PYTHONPATH="${ROOT_DIR}" \
  FLORENCE_MODEL_ID="${MODEL_ID}" \
  FLORENCE_PORT=8081 \
  python3 -m uvicorn integration.api_server:app --host 0.0.0.0 --port 8081 --workers 1 > /tmp/florence_cluster/worker_8081.log 2>&1 &
PID_W1=$!
echo "       Worker PID: ${PID_W1}"

# 2. Start Load Balancer (:8080)
echo "[INFO] Starting Intelligent Load Balancer on port 8080..."
nohup env \
  PYTHONPATH="${ROOT_DIR}" \
  CLUSTER_WORKERS="Florence2-GPU:http://127.0.0.1:8081:gpu" \
  python3 -m uvicorn integration.load_balancer:app --host 0.0.0.0 --port 8080 --workers 1 > /tmp/florence_cluster/load_balancer.log 2>&1 &
PID_LB=$!
echo "       Load Balancer PID: ${PID_LB}"

# Disown so they survive subshell exit
disown -a 2>/dev/null || true

echo "================================================================="
echo "  CLUSTER LAUNCHED SUCCESSFULLY!                                 "
echo "  - Load Balancer : http://localhost:8080                        "
echo "  - Worker 1      : http://localhost:8081 (Florence-2)           "
echo "  - Model         : ${MODEL_ID}                                  "
echo "  - Logs          : /tmp/florence_cluster/                        "
echo "================================================================="
