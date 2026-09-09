# AERIS-3D
## Active Environmental Reasoning & Inverse Simulation for Single-Image 3D Reconstruction

**Smart India Hackathon 2026 — Problem SIH26175 — DepthWizard**
*Organization: ISRO | Category: Software*

---

## Overview

AERIS-3D is a prototype system for **single-view height estimation and 3D flythrough** from a single remote-sensing image. Rather than blindly trusting monocular depth estimation, AERIS treats the depth output as an **initial geometric hypothesis** and actively tries to disprove it through counterfactual testing, hypothesis rejection, and evidence-weighted scoring.

```
INPUT IMAGE
  → IMAGE ANALYSIS
  → INITIAL MONOCULAR DEPTH         (Depth Anything V2 / MiDaS)
  → STRUCTURAL ANALYSIS             (SAM2 / classical CV)
  → MULTIPLE HEIGHT HYPOTHESES      (Candidate Generator)
  → COUNTERFACTUAL TESTING          (CHF + ISCL)
  → PHYSICAL/GEOMETRIC CONSISTENCY  (Shadow, Occlusion, Terrain)
  → HYPOTHESIS REJECTION            (SDRL)
  → BEST-SUPPORTED GEOMETRY         (EGSS)
  → UNCERTAINTY                     (PDU)
  → EVIDENCE-CARRYING DSM/rDSM      (ECDSM)
  → 3D TERRAIN MESH                 (trimesh → GLB)
  → INTERACTIVE FLYTHROUGH          (React + Three.js)
```

---

## Why AERIS is Different

Standard single-image depth pipelines predict depth and stop. AERIS adds a **proposed differentiated algorithmic architecture**:

1. **Hypothesis generation**: Multiple candidate heights are generated per detected structure (not just the depth model's single prediction).
2. **Counterfactual testing (CHF + ISCL)**: Each candidate is projected back into image space and compared against the observed image — edge overlap, segmentation boundary agreement, and depth consistency.
3. **Contradiction detection**: Wrong candidates are actively rejected with a logged reason; survivors move forward.
4. **Self-disproof loop (SDRL)**: Iterative refinement — the system keeps trying to disprove its own best answer until one hypothesis consistently survives.
5. **Evidence-adaptive scoring (EGSS)**: Weights are dynamically adjusted based on data quality — no shadow → shadow weight → 0; weak segmentation → segmentation evidence down-weighted; anchored elevation → terrain evidence up-weighted.
6. **Perturbation-based uncertainty (PDU)**: Small perturbations test how stable the winning estimate is — higher instability → higher reported uncertainty.

---

## Algorithm Glossary

| Acronym | Meaning |
|---|---|
| HCDC | Hypothesis-Conditioned Depth Correction |
| CHF | Counterfactual Height Fingerprinting |
| ISCL | Inverse Sensor Challenge Loop |
| SCT | Shadow Contradiction Testing |
| OOF | Occlusion-Order Falsification |
| TBCS | Terrain-Baseline Contradiction Solver |
| EGSS | Evidence-Weighted Geometric Survival Score |
| SDRL | Self-Disproving Reconstruction Loop |
| PDU | Perturbation-Derived Uncertainty |
| ECDSM | Evidence-Carrying DSM |

---

## Architecture

```
aeris3d/
├── backend/        FastAPI REST API
├── core/           Pipeline modules (depth, structure, HCDC, CHF, …)
├── frontend/       React + Three.js interactive viewer
├── configs/        default.yaml — all tunable parameters
├── data/           input/, cache/, processed/, outputs/
├── tests/          pytest unit + integration tests
├── scripts/        demo.py CLI
├── docs/           Architecture docs
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11+ (tested on 3.13.5)
- Node.js 18+ (tested on v24.13.0)
- Apple Silicon (MPS) or NVIDIA GPU (CUDA) or CPU-only

### Python dependencies

```bash
# Clone and navigate to repo
cd ARIES

# Install Python requirements
pip install -r requirements.txt

# Install PyTorch (Apple Silicon / MPS)
pip install torch torchvision

# Install Depth Anything V2 (downloads on first run via HuggingFace hub)
pip install transformers timm
```

### Frontend dependencies (Phase 18+)

```bash
cd frontend
npm install
```

---

## Usage

### Demo CLI (Gates 1–2)

```bash
# Run on a real image
python scripts/demo.py --input data/input/sample.jpg

# Run with auto-generated synthetic test image
python scripts/demo.py

# Force CPU (disable GPU)
CUDA_VISIBLE_DEVICES="" python scripts/demo.py --input data/input/sample.jpg

# Force re-inference (disable cache)
python scripts/demo.py --input data/input/sample.tif --no-cache
```

Outputs in `data/outputs/`:

| File | Description |
|---|---|
| `input_preview.png` | Resized RGB input |
| `depth.png` | Colorized depth (inferno) |
| `initial_depth.png` | Depth overlay on input |
| `results.json` | Full pipeline metadata |

### Backend (Phase 17+)

```bash
python -m backend.main
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend (Phase 18+)

```bash
cd frontend
npm run dev
# UI at http://localhost:5173
```

---

## Configuration

All tunable parameters are in [`configs/default.yaml`](configs/default.yaml). Key settings:

```yaml
depth:
  model: "depth_anything_v2_small"   # or _base, _large, midas_dpt, gradient_stub
  cache_enabled: true

candidate_generator:
  min_height_m: 3.0
  max_height_m: 60.0
  step_m: 3.0
  top_k: 5

scoring:
  rejection_threshold: 0.35
```

No source code changes needed to switch models, adjust thresholds, or change output paths.

---

## Depth Model Notes

AERIS-3D uses **Depth Anything V2** (Hugging Face) as the primary depth backbone. **Critical domain shift note**: DA-V2 and MiDaS are trained primarily on ground-level, driving, and indoor scenes. Nadir (top-down) aerial/satellite imagery represents a significant distribution shift. Depth output for such imagery is treated as **RELATIVE structural cues only**, not metric elevation, and is labeled accordingly throughout the system.

Metric scale anchoring is possible via:
- GeoTIFF pixel size / GSD (when available)
- Reference DEM (CartoDEM/SRTM/Copernicus, when coverage exists)

---

## Scientific Honesty

- Depth outputs are labeled **RELATIVE** (not metric) unless a real scale source anchors them.
- Terrain is labeled **RELATIVE** or **ANCHORED_METRIC** — never blurred.
- Quantitative metrics (RMSE, MAE) are only computed when real reference data exists.
- No number displayed in the UI is fabricated — all values come from actual pipeline runs.
- Flood visualization is labeled **"SCENARIO SIMULATION — NOT A FLOOD FORECAST."**

---

## Hardware Support

| Environment | Depth Backbone | Status |
|---|---|---|
| Apple Silicon (MPS) | Depth Anything V2 | ✅ Tested |
| NVIDIA CUDA | Depth Anything V2 + mixed precision | ✅ Supported |
| CPU-only | Depth Anything V2 (slow) / gradient_stub | ✅ Fallback |

---

## Feature Status

| Feature | Status |
|---|---|
| Input manager (JPG/PNG/GeoTIFF) | ✅ IMPLEMENTED |
| Monocular depth (DA-V2 / MiDaS) | ✅ IMPLEMENTED |
| Hardware auto-detection | ✅ IMPLEMENTED |
| Depth caching | ✅ IMPLEMENTED |
| Structural analysis (HCDC) | 🔵 PHASE 3+ |
| Candidate generation (CHF) | 🔵 PHASE 5+ |
| Self-disproof (SDRL) | 🔵 PHASE 12+ |
| FastAPI backend | 🔵 PHASE 17+ |
| React + Three.js frontend | 🔵 PHASE 18+ |

---

## Known Limitations (Phase 0–2)

- Depth output is relative only; metric accuracy is not claimed without an elevation anchor.
- No segmentation or candidate generation yet (Prompt 2 scope).
- Frontend not built yet (Prompt 3 scope).
- GeoTIFF support requires `rasterio`; if unavailable, TIF files are rejected cleanly.

---

## SIH Problem Mapping

| SIH Requirement | AERIS-3D Implementation |
|---|---|
| Single-view height estimation | CHF + EGSS (Prompt 2) |
| 3D flythrough | React + Three.js viewer (Prompt 3) |
| ISRO relevance | CartoDEM/Bhuvan anchoring in TBCS |
| Scientific rigor | SDRL + PDU + evidence-adaptive scoring |
| No ground truth fabrication | Strict relative/metric labeling throughout |
