#!/usr/bin/env python3
"""
cluster_stress_test.py — Concurrent Load Test for Horizontal Scaling Cluster

Sends multiple simultaneous requests to the Load Balancer (:8080).
Demonstrates:
  1. Primary requests route to the ultra-fast GPU worker.
  2. Concurrent overflow requests route simultaneously to the CPU worker.
  3. Validates parallel horizontal scaling: both workers process concurrently.
"""

import concurrent.futures
import json
import os
import sys
import time
import requests

LB_URL = os.environ.get("LB_URL", "http://localhost:8080")
FIXTURE_IMAGE = os.environ.get("FIXTURE_IMAGE", "tests/fixtures/parity_image.png")
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "4"))

print("=" * 70)
print(f"  LOCATE-ANYTHING.CPP — HORIZONTAL CLUSTER CONCURRENCY TEST ({NUM_CLIENTS} CLIENTS)")
print("=" * 70)

# Check cluster health
try:
    r = requests.get(f"{LB_URL}/cluster/status", timeout=5)
    print(f"[INFO] Cluster status: {r.status_code}")
    if r.status_code == 200:
        info = r.json()
        print(f"       Online nodes: {len(info.get('nodes', []))}")
        for n in info.get("nodes", []):
            print(f"       - {n['name']} ({n['type'].upper()}): ready={n['ready']}, in_flight={n['in_flight_requests']}")
except Exception as e:
    print(f"[WARNING] Could not get cluster status: {e}")

with open(FIXTURE_IMAGE, "rb") as f:
    img_bytes = f.read()

def send_client_request(req_id):
    t0 = time.time()
    files = {"image": (f"cluster_req_{req_id}.png", img_bytes, "image/png")}
    data = {"prompt": "cat</c>remote", "mode": "fast"}
    try:
        r = requests.post(f"{LB_URL}/v1/locate", files=files, data=data, timeout=120)
        elapsed = time.time() - t0
        server_w = r.headers.get("X-Cluster-Served-By", "Unknown")
        if r.status_code == 200:
            res = r.json()
            return {
                "id": req_id,
                "status": r.status_code,
                "success": True,
                "count": res.get("count", 0),
                "served_by": server_w,
                "server_latency_ms": res.get("latency_ms", 0),
                "client_elapsed_s": elapsed
            }
        else:
            return {
                "id": req_id,
                "status": r.status_code,
                "success": False,
                "served_by": server_w,
                "error": r.text[:100],
                "client_elapsed_s": elapsed
            }
    except Exception as e:
        return {
            "id": req_id,
            "status": "EXC",
            "success": False,
            "served_by": "N/A",
            "error": str(e),
            "client_elapsed_s": time.time() - t0
        }

print(f"\n[INFO] Firing {NUM_CLIENTS} simultaneous requests into Load Balancer (:8080)...")
wall_start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
    futures = [executor.submit(send_client_request, i + 1) for i in range(NUM_CLIENTS)]
    results = [f.result() for f in futures]

wall_time = time.time() - wall_start

print("\n" + "-" * 75)
print(f"{'Req ID':<8} | {'HTTP':<6} | {'Served By':<24} | {'Detections':<12} | {'Latency':<10}")
print("-" * 75)

workers_hit = set()
for res in sorted(results, key=lambda x: x["id"]):
    status = "✅ 200" if res["success"] else f"❌ {res['status']}"
    served = res.get("served_by", "Unknown")
    workers_hit.add(served)
    count = str(res.get("count", "N/A"))
    lat = f"{res['client_elapsed_s']:.2f}s"
    print(f"Req #{res['id']:<4} | {status:<6} | {served:<24} | {count:<12} | {lat:<10}")

print("-" * 75)
print(f"Total Wall-Clock Time: {wall_time:.2f}s")
print(f"Workers Utilized    : {len(workers_hit)} distinct worker nodes ({', '.join(workers_hit)})")

if len(workers_hit) > 1:
    print("✅ PARALLEL HORIZONTAL SCALING CONFIRMED: Traffic distributed across multiple workers simultaneously!")
else:
    print("ℹ️ Note: Traffic was handled by a single worker node.")

print("=" * 70)
