"""
load_balancer.py — Intelligent Reverse Proxy Load Balancer for LocateAnything Cluster

Routes inference traffic across a pool of backend workers:
  - Primary: GPU Worker (port 8081, high-throughput, ~0.8s/img)
  - Secondary/Overflow: CPU Worker(s) (port 8082+, parallel processing)

Features:
  1. Priority / Weighted Dispatching: Prioritizes idle GPU workers first;
     overflows concurrent traffic to CPU workers to prevent queuing.
  2. Automatic Health Checks: Periodic background polling of worker /ready endpoints.
  3. Seamless Multipart Forwarding: Passes image files, prompts, and options transparently.
  4. Real-time Cluster Telemetry: GET /cluster/status returns detailed metrics.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib.error
import urllib.request

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CLUSTER-LB] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("load_balancer")

app = FastAPI(
    title="LocateAnything Horizontal Scaling Cluster",
    description="Intelligent Reverse-Proxy Load Balancer routing between CUDA GPU and CPU worker instances.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parse worker configuration from environment
# Format: "name:url:type,name:url:type"
# e.g.: "GPU-Worker:http://127.0.0.1:8081:gpu,CPU-Worker:http://127.0.0.1:8082:cpu"
DEFAULT_WORKERS_CONF = (
    "GPU-Worker-1:http://127.0.0.1:8081:gpu,"
    "CPU-Worker-1:http://127.0.0.1:8082:cpu"
)
WORKERS_CONF = os.environ.get("CLUSTER_WORKERS", DEFAULT_WORKERS_CONF)

class WorkerNode:
    def __init__(self, name: str, url: str, worker_type: str):
        self.name = name
        self.url = url.rstrip("/")
        self.type = worker_type  # 'gpu' or 'cpu'
        self.is_ready = False
        self.in_flight = 0
        self.total_served = 0
        self.total_failures = 0
        self.avg_latency_ms = 0.0
        self.last_check = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "type": self.type,
            "ready": self.is_ready,
            "in_flight_requests": self.in_flight,
            "total_served": self.total_served,
            "total_failures": self.total_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1)
        }

WORKERS: List[WorkerNode] = []
for entry in WORKERS_CONF.split(","):
    entry = entry.strip()
    if not entry:
        continue
    parts = entry.split(":")
    if len(parts) >= 5:
        # name:http://host:port:type
        w_name = parts[0]
        w_url = f"{parts[1]}:{parts[2]}:{parts[3]}"
        w_type = parts[4]
    elif len(parts) == 4:
        # name:http://host:port
        w_name = parts[0]
        w_url = f"{parts[1]}:{parts[2]}:{parts[3]}"
        w_type = "gpu" if "gpu" in w_name.lower() else "cpu"
    else:
        w_name = parts[0]
        w_url = "http://127.0.0.1:8081"
        w_type = "gpu"
    WORKERS.append(WorkerNode(w_name, w_url, w_type))

logger.info(f"Initialized cluster with {len(WORKERS)} worker nodes: {[w.to_dict() for w in WORKERS]}")

async def poll_worker_health():
    """Background task checking worker readiness periodically."""
    while True:
        for worker in WORKERS:
            # If worker is currently executing an in-flight request, it is actively alive
            if worker.in_flight > 0:
                worker.is_ready = True
                continue

            try:
                loop = asyncio.get_event_loop()
                req = urllib.request.Request(f"{worker.url}/ready", method="GET")

                def do_check():
                    try:
                        with urllib.request.urlopen(req, timeout=5.0) as resp:
                            return resp.status == 200
                    except Exception:
                        return False

                ready = await loop.run_in_executor(None, do_check)
                if ready != worker.is_ready:
                    status_text = "ONLINE (Ready)" if ready else "OFFLINE (Unreachable)"
                    logger.info(f"Worker [{worker.name}] ({worker.type.upper()}) changed status to: {status_text}")
                worker.is_ready = ready
                worker.last_check = time.time()
            except Exception as e:
                worker.is_ready = False
        await asyncio.sleep(3.0)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(poll_worker_health())
    logger.info("Load balancer startup completed. Health monitoring active.")

def select_best_worker() -> Optional[WorkerNode]:
    """
    Selects the optimal worker:
      1. Idle GPU worker (in_flight == 0)
      2. Idle CPU worker (in_flight == 0) for true horizontal overflow
      3. Worker with the minimum queue depth
    """
    online_workers = [w for w in WORKERS if w.is_ready]
    if not online_workers:
        # Fallback: if health check hasn't run yet, try any worker
        online_workers = WORKERS

    if not online_workers:
        return None

    # Step 1: Check for idle GPU worker
    idle_gpus = [w for w in online_workers if w.type == "gpu" and w.in_flight == 0]
    if idle_gpus:
        return idle_gpus[0]

    # Step 2: Check for idle CPU worker (parallel overflow)
    idle_cpus = [w for w in online_workers if w.type == "cpu" and w.in_flight == 0]
    if idle_cpus:
        return idle_cpus[0]

    # Step 3: Pick worker with minimum in_flight requests
    return min(online_workers, key=lambda w: w.in_flight)

async def forward_request(worker: WorkerNode, path: str, request: Request) -> Response:
    """Forwards incoming client HTTP request directly to target worker."""
    target_url = f"{worker.url}{path}"
    body = await request.body()
    headers = dict(request.headers)
    # Remove host header to avoid reverse proxy confusion
    headers.pop("host", None)
    headers.pop("content-length", None)

    worker.in_flight += 1
    t0 = time.time()
    loop = asyncio.get_event_loop()

    def do_forward():
        req = urllib.request.Request(target_url, data=body, headers=headers, method=request.method)
        try:
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                resp_data = resp.read()
                resp_headers = dict(resp.headers)
                return resp.status, resp_headers, resp_data
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except Exception as e:
            return 502, {}, json.dumps({"detail": f"Gateway error connecting to worker: {e}"}).encode()

    try:
        status, resp_headers, resp_data = await loop.run_in_executor(None, do_forward)
        elapsed_ms = (time.time() - t0) * 1000.0
        worker.total_served += 1
        # Exponential moving average
        worker.avg_latency_ms = (
            elapsed_ms if worker.avg_latency_ms == 0 else (0.8 * worker.avg_latency_ms + 0.2 * elapsed_ms)
        )
        
        # Strip content encoding if present
        resp_headers.pop("content-encoding", None)
        resp_headers.pop("transfer-encoding", None)
        resp_headers["X-Cluster-Served-By"] = f"{worker.name} ({worker.type.upper()})"
        resp_headers["X-Cluster-Worker-Url"] = worker.url

        content_type = resp_headers.get("content-type", "application/json")
        return Response(content=resp_data, status_code=status, headers=resp_headers, media_type=content_type)
    except Exception as e:
        worker.total_failures += 1
        raise HTTPException(status_code=502, detail=f"Failed forwarding to {worker.name}: {e}")
    finally:
        worker.in_flight = max(0, worker.in_flight - 1)

@app.get("/healthz")
async def healthz():
    online = [w.name for w in WORKERS if w.is_ready]
    return {
        "status": "alive",
        "cluster_online_workers": len(online),
        "total_workers": len(WORKERS),
        "workers": [w.to_dict() for w in WORKERS]
    }

@app.get("/ready")
async def ready():
    online = [w for w in WORKERS if w.is_ready]
    if online:
        return {"status": "ready", "available_workers": len(online)}
    raise HTTPException(status_code=503, detail="No cluster workers currently ready.")

@app.get("/cluster/status")
async def cluster_status():
    total_served = sum(w.total_served for w in WORKERS)
    total_in_flight = sum(w.in_flight for w in WORKERS)
    return {
        "cluster_name": "LocateAnything-Scale-Cluster",
        "total_served_requests": total_served,
        "total_in_flight": total_in_flight,
        "worker_count": len(WORKERS),
        "nodes": [w.to_dict() for w in WORKERS]
    }

@app.post("/v1/locate")
async def locate_proxy(request: Request):
    worker = select_best_worker()
    if not worker:
        raise HTTPException(status_code=503, detail="No active backend workers available.")
    logger.info(f"Dispatching POST /v1/locate -> {worker.name} (type: {worker.type}, in_flight: {worker.in_flight})")
    return await forward_request(worker, "/v1/locate", request)

@app.post("/v1/locate/annotated")
async def locate_annotated_proxy(request: Request):
    worker = select_best_worker()
    if not worker:
        raise HTTPException(status_code=503, detail="No active backend workers available.")
    logger.info(f"Dispatching POST /v1/locate/annotated -> {worker.name} (type: {worker.type}, in_flight: {worker.in_flight})")
    return await forward_request(worker, "/v1/locate/annotated", request)

# ---------------------------------------------------------------------------
# Static Web Dashboard & Sample Assets
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
SAMPLES_DIR = Path(__file__).parent.parent / "benchmarks" / "media"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_dashboard():
    """Serve the Web Dashboard Studio at root URL."""
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return JSONResponse({"status": "ready", "ui": "Web Dashboard not found in static directory."})

