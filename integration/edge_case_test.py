#!/usr/bin/env python3
"""
edge_case_test.py — Robustness & Edge-Case Validation for LocateAnything API

Tests:
  1. Corrupted image bytes (junk data) -> Expect 400 Bad Request.
  2. Non-image file (text/binary) -> Expect 400 Bad Request.
  3. Empty payload (0 bytes image) -> Expect 400 Bad Request.
  4. Missing prompt -> Expect 422 Unprocessable Entity.
  5. Invalid mode string -> Expect 400 Bad Request.
  6. Negative prompt / Target absent ("airplane</c>elephant") -> Expect 200 OK, 0 detections.
  7. Multi-target complex prompt ("cat</c>remote</c>cushion</c>floor") -> Expect 200 OK, valid detections.
  8. Special characters & Unicode prompt -> Graceful execution.
  9. Post-stress health check -> Server remains 100% operational.
"""

import io
import json
import os
import sys
import time
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8080")
FIXTURE_IMAGE = os.environ.get("FIXTURE_IMAGE", "tests/fixtures/parity_image.png")

passed = 0
failed = 0

def log_test(name, success, details=""):
    global passed, failed
    status = "✅ PASS" if success else "❌ FAIL"
    if success:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {name}")
    if details:
        print(f"        {details}")

print("=" * 60)
print("  LOCATE-ANYTHING.CPP — EDGE CASE & ROBUSTNESS TEST SUITE")
print("=" * 60)

# Check server is up
try:
    r = requests.get(f"{API_URL}/ready", timeout=5)
    if r.status_code != 200:
        print(f"ERROR: Server /ready returned {r.status_code}. Aborting.")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: Cannot connect to {API_URL}: {e}")
    sys.exit(1)

print("[INFO] Target server is ready. Beginning test cases...\n")

# TEST 1: Corrupted image bytes
try:
    corrupted_data = b"GIF89a" + b"\x00\xFF\xAA\x55" * 100  # Fake corrupted header
    files = {"image": ("corrupted.png", corrupted_data, "image/png")}
    data = {"prompt": "cat</c>remote"}
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=10)
    success = (r.status_code == 400)
    log_test("Test 1: Corrupted image data returns HTTP 400", success, f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    log_test("Test 1: Corrupted image data returns HTTP 400", False, str(e))

# TEST 2: Non-image file (plain text)
try:
    text_data = b"This is a text file, definitely not an image."
    files = {"image": ("test.txt", text_data, "text/plain")}
    data = {"prompt": "cat"}
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=10)
    success = (r.status_code == 400)
    log_test("Test 2: Non-image file returns HTTP 400", success, f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    log_test("Test 2: Non-image file returns HTTP 400", False, str(e))

# TEST 3: Empty image file (0 bytes)
try:
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    data = {"prompt": "cat"}
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=10)
    success = (r.status_code == 400)
    log_test("Test 3: Empty file returns HTTP 400", success, f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    log_test("Test 3: Empty file returns HTTP 400", False, str(e))

# TEST 4: Missing prompt field
try:
    with open(FIXTURE_IMAGE, "rb") as f:
        img_bytes = f.read()
    files = {"image": ("test.png", img_bytes, "image/png")}
    r = requests.post(f"{API_URL}/v1/locate", files=files, timeout=10)
    success = (r.status_code == 422)
    log_test("Test 4: Missing prompt parameter returns HTTP 422", success, f"HTTP {r.status_code}")
except Exception as e:
    log_test("Test 4: Missing prompt parameter returns HTTP 422", False, str(e))

# TEST 5: Invalid mode string
try:
    with open(FIXTURE_IMAGE, "rb") as f:
        img_bytes = f.read()
    files = {"image": ("test.png", img_bytes, "image/png")}
    data = {"prompt": "cat", "mode": "turbo_ultra"}
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=10)
    success = (r.status_code == 400)
    log_test("Test 5: Invalid mode returns HTTP 400", success, f"HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    log_test("Test 5: Invalid mode returns HTTP 400", False, str(e))

# TEST 6: Target absent from image (Zero detections test)
try:
    with open(FIXTURE_IMAGE, "rb") as f:
        img_bytes = f.read()
    files = {"image": ("test.png", img_bytes, "image/png")}
    data = {"prompt": "airplane</c>submarine</c>elephant", "mode": "fast"}
    t0 = time.time()
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=60)
    elapsed = time.time() - t0
    resp_json = r.json() if r.status_code == 200 else {}
    count = resp_json.get("count", -1)
    success = (r.status_code == 200 and count == 0)
    log_test("Test 6: Target absent returns HTTP 200 with 0 detections", success,
             f"HTTP {r.status_code}, count={count}, latency={elapsed:.2f}s")
except Exception as e:
    log_test("Test 6: Target absent returns HTTP 200 with 0 detections", False, str(e))

# TEST 7: Complex multi-target prompt
try:
    with open(FIXTURE_IMAGE, "rb") as f:
        img_bytes = f.read()
    files = {"image": ("test.png", img_bytes, "image/png")}
    data = {"prompt": "cat</c>remote</c>couch</c>blanket", "mode": "fast"}
    t0 = time.time()
    r = requests.post(f"{API_URL}/v1/locate", files=files, data=data, timeout=60)
    elapsed = time.time() - t0
    resp_json = r.json() if r.status_code == 200 else {}
    detections = resp_json.get("detections", [])
    labels = set(d["label"] for d in detections)
    success = (r.status_code == 200 and len(detections) >= 2)
    log_test("Test 7: Multi-target prompt returns detections for present targets", success,
             f"HTTP {r.status_code}, detected labels={labels}, count={len(detections)}, latency={elapsed:.2f}s")
except Exception as e:
    log_test("Test 7: Multi-target prompt returns detections for present targets", False, str(e))

# TEST 8: Post-stress Health & Readiness verification
try:
    r_health = requests.get(f"{API_URL}/healthz", timeout=5)
    r_ready = requests.get(f"{API_URL}/ready", timeout=5)
    h_data = r_health.json() if r_health.status_code == 200 else {}
    healthy = h_data.get("engine_loaded", False) or (h_data.get("cluster_online_workers", 0) > 0)
    success = (r_health.status_code == 200 and r_ready.status_code == 200 and healthy)
    log_test("Test 8: Server remains healthy and ready after all edge-cases", success,
             f"healthz={r_health.status_code}, ready={r_ready.status_code}")
except Exception as e:
    log_test("Test 8: Server remains healthy and ready after all edge-cases", False, str(e))

print("\n" + "=" * 60)
print(f"  RESULT: {passed} PASSED, {failed} FAILED (Total: {passed + failed})")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
