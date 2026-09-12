#!/usr/bin/env python3
"""
example_usage.py — Demonstration of FlorenceClient SDK
"""
import sys
from pathlib import Path

# Add sdk dir to sys.path
sys.path.insert(0, str(Path(__file__).parent))
from florence_client import FlorenceClient

def main():
    client = FlorenceClient("http://localhost:8080")

    print("[1] Checking service readiness...")
    ready = client.is_ready()
    print(f"    Service is ready: {ready}")
    if not ready:
        print("Service is not ready, exiting.")
        return

    test_img = Path(__file__).parent.parent / "tests" / "fixtures" / "parity_image.png"
    if not test_img.is_file():
        print(f"Test image not found at {test_img}")
        return

    print(f"\n[2] Running Phrase Grounding on {test_img.name} (prompt: 'cat')...")
    res = client.locate(test_img, prompt="cat")
    print(f"    Latency: {res.latency_ms} ms")
    print(f"    Detections found ({res.count}):")
    for d in res.detections:
        print(f"      - {d.label} at {d.box}")

    print(f"\n[3] Running Detect All (<OD>)...")
    res_od = client.detect_all(test_img)
    print(f"    Latency: {res_od.latency_ms} ms | Count: {res_od.count}")

    print(f"\n[4] Running OCR on {test_img.name}...")
    res_ocr = client.ocr(test_img)
    print(f"    OCR Latency: {res_ocr.latency_ms} ms | Text regions: {res_ocr.count}")
    if res_ocr.full_text:
        print(f"    Extracted Text: \"{res_ocr.full_text}\"")

    out_png = Path(__file__).parent / "sdk_output.png"
    print(f"\n[5] Downloading annotated image to {out_png.name}...")
    saved_path = client.download_annotated(test_img, "cat", out_png)
    print(f"    Saved: {saved_path} ({saved_path.stat().st_size} bytes)")

    print("\n✅ SDK Demonstration Completed Successfully!")

if __name__ == "__main__":
    main()
