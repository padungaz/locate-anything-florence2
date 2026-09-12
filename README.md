# Florence-2 Vision Grounding Service (Commercial Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Microsoft Florence-2](https://img.shields.io/badge/Model-Florence--2--large-blue)](https://huggingface.co/microsoft/Florence-2-large)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-teal)](https://fastapi.tiangolo.com/)

An open-vocabulary visual grounding, object detection, and cluster-scale inference service powered by **Microsoft Florence-2-large**.

> [!IMPORTANT]
> **100% Commercial License (MIT)**: Both the backend code and the Microsoft Florence-2 model weights are released under the **permissive MIT License**. This repository contains no non-commercial restrictions and is fully cleared for commercial products, SaaS deployment, and private cloud hosting.

---

## Features

- **Open-Vocabulary Visual Grounding**: Detect any object or phrase described in natural language (e.g. `cat</c>remote`, `person wearing a helmet`).
- **High-Accuracy Sequence-to-Sequence Vision Foundation**: Powered by `microsoft/Florence-2-large` (776M parameters) fine-tuned on FLD-5B annotations.
- **Intelligent Load Balancing & Horizontal Cluster**: Built-in weighted round-robin and health-monitoring load balancer (`integration/load_balancer.py`).
- **Interactive Web Studio Dashboard**: Modern HTML5 Canvas interface with real-time bounding box inspection, cluster metrics, and image drag-and-drop.
- **Production REST API**: Standardized JSON response (`POST /v1/locate`) and annotated image rendering (`POST /v1/locate/annotated`).
- **High Reliability & Edge-Case Protection**: Automated filters to eliminate full-frame false positives on absent objects, strict payload validation, and full resilience against corrupted inputs.

---

## Quickstart

### 1. Prerequisites & Installation

```bash
# Clone and enter directory
cd locate-anything-florence2

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Florence-2-large Model Weights

```bash
python3 scripts/download_florence2.py
```

### 3. Start the Inference Cluster

```bash
# Launch worker (:8081) and Load Balancer (:8080)
bash scripts/start_cluster.sh
```

The services will be available at:
- **Web Studio & Cluster Dashboard**: [http://localhost:8080](http://localhost:8080)
- **Primary Worker**: [http://localhost:8081](http://localhost:8081)
- **API Documentation (Swagger)**: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## API Reference

### `POST /v1/locate`
Detect objects and return bounding boxes in pixel coordinates.

```bash
curl -X POST http://localhost:8080/v1/locate \
  -F "image=@photo.jpg" \
  -F "prompt=cat</c>remote" \
  -F "mode=grounding"
```

**Response:**
```json
{
  "detections": [
    {
      "label": "cat",
      "box": [6.944, 49.056, 221.536, 441.504]
    }
  ],
  "count": 1,
  "latency_ms": 420.5,
  "mode": "grounding"
}
```

### `POST /v1/locate/annotated`
Returns the input image with bounding boxes and labels drawn directly onto the image.

```bash
curl -X POST http://localhost:8080/v1/locate/annotated \
  -F "image=@photo.jpg" \
  -F "prompt=cat</c>remote" \
  -o annotated_output.png
```

### Health & Monitoring
- `GET /healthz`: Process liveness probe
- `GET /ready`: Model readiness check
- `GET /cluster/status`: Real-time node status, active requests, and latency statistics

---

## Automated Test Suite

Run the comprehensive test suite (cluster verification, REST endpoints, detection accuracy, edge-case robustness, annotated rendering):

```bash
bash scripts/verify_all.sh
```

---

## License

This project is licensed under the **MIT License**.

- **Software Code**: MIT License (see [LICENSE](LICENSE))
- **Model Weights**: Microsoft Florence-2-large is distributed under the MIT License by Microsoft (see [HuggingFace Model Card](https://huggingface.co/microsoft/Florence-2-large)).
