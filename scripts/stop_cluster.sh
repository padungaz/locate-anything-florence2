#!/bin/bash
# ==============================================================================
# stop_cluster.sh — Gracefully shutdown all LocateAnything cluster services
# ==============================================================================
echo "Stopping LocateAnything cluster processes on ports 8080, 8081, 8082..."
fuser -k 8080/tcp 8081/tcp 8082/tcp 2>/dev/null || true
sleep 1
echo "All cluster services stopped."
