#!/bin/bash
# ==============================================================================
# run_all_tests.sh — Run Master QA Test Suite against Active Cluster
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"

LB_URL="${LB_URL:-http://localhost:8080}"

echo "================================================================="
echo "  LOCATE-ANYTHING.CPP — MASTER QUALITY ASSURANCE TEST SUITE      "
echo "  Target URL: ${LB_URL}                                         "
echo "================================================================="

# Check cluster readiness
echo -e "\n>>> 1/3: Checking Cluster Readiness..."
curl -s "${LB_URL}/ready" || {
    echo "[ERROR] Cluster at ${LB_URL} is not ready. Please start it with ./scripts/start_cluster.sh"
    exit 1
}
echo " Cluster is READY!"

echo -e "\n>>> 2/3: Running Robustness & Edge-Cases Test (edge_case_test.py)..."
python3 integration/edge_case_test.py

echo -e "\n>>> 3/3: Running Cluster Multi-Client Concurrency Test (cluster_stress_test.py)..."
NUM_CLIENTS=3 python3 integration/cluster_stress_test.py

echo -e "\n================================================================="
echo "  ✅ ALL CLUSTER TESTS COMPLETED & PASSED WITH SUCCESS!          "
echo "================================================================="
