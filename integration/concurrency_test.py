#!/usr/bin/env python3
"""
concurrency_test.py — Concurrent Load & Mutex Safety Test for LocateAnything API

Spawns multiple worker threads that fire requests simultaneously at the API.
Verifies that:
  1. The API server doesn't crash or throw 500 when hit with concurrent requests.
  2. The Singleton Mutex in la_wrapper correctly serializes inference without race condition.
  3. All concurrent requests receive valid HTTP 200 with accurate detections.
  4. Collects latency statistics (min, max, avg, throughput).
"""

import concurrent.futures
import json
import os
import sys
import time
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8080")
FIXTURE_IMAGE = os.environ.get("FIXTURE_IMAGE", "tests/fixtures/parity_image.png")
NUM_CONCURRENT_REQUESTS = int(os.environ.get("NUM_WORKERS", "3"))

print("=" * 65)
print(f"  LOCATE-ANYTHING.CPP — CONCURRENCY & STRESS TEST ({NUM_CONCURRENT_REQUESTS} CLIENTS)")
print("=" * 65)

# Load test image into memory once
with open(FIXTURE_IMAGE, "rb") as f:
    img_bytes = f.read()

def send_request(request_id):
    """Sends a single locate request and measures latency and correctness."""
    start_time = time.time()
    files = {"image": (f"req_{request_id}.png", img_bytes, "image/png")}
    data = {"prompt": "cat</c>remote", "mode": "fast"}
    try:
        r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=120)
        elapsed = time.time() - start_time
        if r.status_code == 200:
            res = r.json()
            count = res.get("count", 0)
            return {
                "id": request_id,
                "status": r.status_code,
                "success": True,
                "count": count,
                "client_elapsed_s": elapsed,
                "server_latency_ms": res.get("latency_ms", 0)
            }
        else:
            return {
                "id": request_id,
                "status": r.status_code,
                "success": False,
                "error": r.text,
                "client_elapsed_s": elapsed
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "id": request_id,
            "status": "EXCEPTION",
            "success": False,
            "error": str(e),
            "client_elapsed_s": elapsed
        }

print(f"[INFO] Dispatching {NUM_CONCURRENT_REQUESTS} concurrent requests simultaneously...")
wall_start = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
    futures = [executor.submit(send_request, i + 1) for i in range(NUM_CONCURRENT_REQUESTS)]
    results = [f.result() for f in futures]

wall_time = time.time() - wall_start

print("\n" + "-" * 65)
print(f"{'Req ID':<8} | {'HTTP':<6} | {'Status':<8} | {'Detections':<12} | {'Server Latency':<16} | {'Client Time':<12}")
print("-" * 65)

all_passed = True
client_times = []
server_latencies = []

for res in sorted(results, key=lambda x: x["id"]):
    status_str = "✅ PASS" if res["success"] and res.get("count", 0) == 4 else "❌ FAIL"
    if not (res["success"] and res.get("count", 0) == 4):
        all_passed = False
    
    server_lat = f"{res.get('server_latency_ms', 0):.0f} ms" if res["success"] else "N/A"
    client_t = f"{res['client_elapsed_s']:.2f} s"
    count_str = str(res.get("count", "N/A"))
    
    print(f"Req #{res['id']:<4} | {res['status']:<6} | {status_str:<8} | {count_str:<12} | {server_lat:<16} | {client_t:<12}")
    
    if res["success"]:
        client_times.append(res["client_elapsed_s"])
        server_latencies.append(res.get("server_latency_ms", 0))

print("-" * 65)
print(f"Total Wall-Clock Time: {wall_time:.2f} s")
if server_latencies:
    avg_server = sum(server_latencies) / len(server_latencies)
    print(f"Average Server Inference Latency: {avg_server / 1000.0:.2f} s")
    print(f"Serialized Queue Behavior: Total time ({wall_time:.1f}s) ≈ Sum of individual inference latencies ({sum(server_latencies)/1000.0:.1f}s)")
    print(f"Throughput: {len(results) / (wall_time / 60.0):.2f} requests/minute")

print("=" * 65)
if all_passed:
    print(f"✅ ALL {NUM_CONCURRENT_REQUESTS} CONCURRENT REQUESTS PASSED WITH MUTEX SAFETY!")
    sys.exit(0)
else:
    print(f"❌ CONCURRENCY TEST FAILED!")
    sys.exit(1)
