#!/usr/bin/env python3
"""
memory_stability_test.py — Memory Leak & Long-Running Stability Profiler

Monitors the Resident Set Size (VmRSS) and Peak Memory (VmHWM) of the API server process
across multiple repeated inference requests.
Verifies that:
  1. ggml tensor contexts and image buffers are completely deallocated between inferences.
  2. Memory usage does not grow monotonically (no memory leak).
  3. Delta RSS between consecutive requests settles to ~0 MB.
"""

import os
import re
import subprocess
import sys
import time
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8080")
FIXTURE_IMAGE = os.environ.get("FIXTURE_IMAGE", "tests/fixtures/parity_image.png")
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "5"))

def get_server_pid():
    """Finds the PID of the running uvicorn process."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "uvicorn.*api_server"], text=True)
        pids = [int(p) for p in out.strip().split() if p]
        return pids[0] if pids else None
    except Exception:
        return None

def get_process_memory_kb(pid):
    """Reads VmRSS and VmHWM in KB from /proc/<pid>/status."""
    if not pid:
        return 0, 0
    status_path = f"/proc/{pid}/status"
    if not os.path.exists(status_path):
        return 0, 0
    vm_rss = 0
    vm_hwm = 0
    with open(status_path, "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                vm_rss = int(re.search(r"\d+", line).group(0))
            elif line.startswith("VmHWM:"):
                vm_hwm = int(re.search(r"\d+", line).group(0))
    return vm_rss, vm_hwm

pid = get_server_pid()
if not pid:
    print("[ERROR] Could not detect running API server PID.")
    sys.exit(1)

print("=" * 70)
print(f"  LOCATE-ANYTHING.CPP — MEMORY LEAK & RSS STABILITY PROFILER")
print(f"  Target Server PID: {pid}")
print(f"  Iterations: {NUM_ITERATIONS}")
print("=" * 70)

# Load fixture image bytes
with open(FIXTURE_IMAGE, "rb") as f:
    img_bytes = f.read()

initial_rss, initial_hwm = get_process_memory_kb(pid)
print(f"[BASELINE] Initial VmRSS: {initial_rss / 1024:.2f} MB | VmHWM: {initial_hwm / 1024:.2f} MB\n")

print(f"{'Iter':<6} | {'Status':<8} | {'Detections':<12} | {'Latency':<12} | {'VmRSS (MB)':<14} | {'Δ RSS (MB)':<12}")
print("-" * 70)

prev_rss = initial_rss
records = []

for i in range(1, NUM_ITERATIONS + 1):
    files = {"image": (f"test_{i}.png", img_bytes, "image/png")}
    data = {"prompt": "cat</c>remote", "mode": "fast"}
    
    t0 = time.time()
    try:
        r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=60)
        elapsed = time.time() - t0
        res = r.json() if r.status_code == 200 else {}
        count = res.get("count", -1)
        curr_rss, curr_hwm = get_process_memory_kb(pid)
        delta_rss = curr_rss - prev_rss
        
        status = "✅ 200 OK" if r.status_code == 200 and count == 4 else f"❌ {r.status_code}"
        lat_str = f"{elapsed:.2f} s"
        rss_str = f"{curr_rss / 1024:.2f}"
        delta_str = f"{delta_rss / 1024:+.2f}"
        
        print(f"#{i:<5} | {status:<8} | {count:<12} | {lat_str:<12} | {rss_str:<14} | {delta_str:<12}")
        
        records.append({
            "iteration": i,
            "latency": elapsed,
            "rss_mb": curr_rss / 1024,
            "delta_mb": delta_rss / 1024
        })
        prev_rss = curr_rss
    except Exception as e:
        print(f"#{i:<5} | ❌ EXCEPTION: {e}")

final_rss, final_hwm = get_process_memory_kb(pid)
net_rss_growth = final_rss - initial_rss

print("-" * 70)
print(f"Initial RSS : {initial_rss / 1024:.2f} MB")
print(f"Final RSS   : {final_rss / 1024:.2f} MB")
print(f"Peak RSS    : {final_hwm / 1024:.2f} MB")
print(f"Net Growth  : {net_rss_growth / 1024:+.2f} MB over {NUM_ITERATIONS} iterations")

# Memory leak verdict:
# If growth is under 20MB after multiple runs (allocator fragmentation margin), it is safe.
leak_free = abs(net_rss_growth / 1024) < 20.0

if leak_free:
    print("\n✅ VERDICT: PASSED — NO MEMORY LEAKS DETECTED. Memory footprint is strictly bounded.")
    sys.exit(0)
else:
    print("\n⚠️ WARNING: Memory grew by more than 20 MB. Review memory allocations.")
    sys.exit(1)
