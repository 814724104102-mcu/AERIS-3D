# AERIS-3D
## Active Environmental Reasoning & Inverse Simulation for Single-Image 3D Reconstruction

**Smart India Hackathon 2026 — Problem SIH26175 — DepthWizard**  
*Organization: ISRO | Category: Software*

---

## Overview

AERIS-3D is a prototype system for **single-view height estimation and interactive 3D flythrough** from a single remote-sensing image. Rather than blindly trusting monocular depth estimation, AERIS treats depth output as an **initial geometric hypothesis** and actively tries to disprove it through counterfactual testing, hypothesis rejection, and evidence-weighted scoring.

```
INPUT IMAGE
  → IMAGE ANALYSIS
  → INITIAL MONOCULAR DEPTH         (Depth Anything V2 / MiDaS / gradient_stub)
  → STRUCTURAL ANALYSIS             (Classical CV: Canny + SLIC + heuristics)
  → MULTIPLE HEIGHT HYPOTHESES      (Candidate Generator per detected building)
  → COUNTERFACTUAL TESTING          (CHF + ISCL)
  → PHYSICAL/GEOMETRIC CONSISTENCY  (Shadow, Occlusion, Terrain)
  → HYPOTHESIS REJECTION            (SDRL — Self-Disproving Reconstruction Loop)
  → BEST-SUPPORTED GEOMETRY         (EGSS — Evidence-Weighted Geometric Survival Score)
  → UNCERTAINTY                     (PDU — Perturbation-Derived Uncertainty)
  → EVIDENCE-CARRYING DSM/rDSM      (ECDSM)
  → 3D TERRAIN MESH                 (trimesh → GLB, vertex-colored with input image)
  → INTERACTIVE FLYTHROUGH          (React + Three.js)
```

---

## Why AERIS is Different

Standard single-image depth pipelines predict depth and stop. AERIS adds a **proposed differentiated algorithmic architecture**:

1. **Hypothesis generation**: Multiple candidate heights per detected structure — not just the depth model's single prediction.
2. **Counterfactual testing (CHF + ISCL)**: Each candidate is projected back into image space and compared against the observed image — edge overlap, segmentation boundary agreement, depth consistency.
3. **Contradiction detection**: Wrong candidates are actively rejected with a logged reason; survivors move forward.
4. **Self-disproof loop (SDRL)**: Iterative refinement — the system keeps trying to disprove its own best answer until one hypothesis consistently survives.
5. **Evidence-adaptive scoring (EGSS)**: Weights are dynamically adjusted based on data quality — no shadow → shadow weight → 0; weak segmentation → segmentation evidence down-weighted; anchored elevation → terrain evidence up-weighted.
6. **Perturbation-based uncertainty (PDU)**: Small perturbations test how stable the winning estimate is — higher instability → higher reported uncertainty.

---

## Algorithm Glossary

| Acronym | Meaning | Role |
|---|---|---|
| HCDC | Hypothesis-Conditioned Depth Correction | Refines raw monocular depth using structure |
| CHF | Counterfactual Height Fingerprinting | Tests a range of candidate heights per object |
| ISCL | Inverse Sensor Challenge Loop | Reprojects candidates into image space, compares to real image |
| SCT | Shadow Contradiction Testing | Uses shadow evidence when reliable |
| OOF | Occlusion-Order Falsification | Checks whether candidate geometry gives plausible front/behind ordering |
| TBCS | Terrain-Baseline Contradiction Solver | Separates object height from terrain elevation; relative→metric calibration |
| EGSS | Evidence-Weighted Geometric Survival Score | Combines all evidence into one adaptive per-candidate score |
| SDRL | Self-Disproving Reconstruction Loop | Iteratively rejects weak candidates until one survives |
| PDU | Perturbation-Derived Uncertainty | Measures how stable the winner is under small perturbations |
| ECDSM | Evidence-Carrying DSM | Final elevation surface tagged with confidence/uncertainty/evidence per point |

---

## SIH Problem Mapping

| SIH26175 Requirement | AERIS-3D Implementation |
|---|---|
| Single-view height estimation | CHF + EGSS (Phases 5–11) |
| 3D flythrough | React + Three.js viewer (Phase 18–20) |
| ISRO relevance | CartoDEM/Bhuvan/SRTM anchoring in TBCS |
| Scientific rigor | SDRL + PDU + evidence-adaptive scoring |
| No ground truth fabrication | Strict RELATIVE/METRIC labeling throughout |
| Slope assessment | TBCS terrain roughness + slope display |
| rDSM (plain image) | ✅ Implemented |
| DSM (GeoTIFF + anchor) | ✅ Implemented (GeoTIFF only when CRS available) |

**Official deliverables mapped to architecture:**
- **Elevation Estimation Module** = `core/` pipeline + FastAPI backend (`backend/`)
- **Interactive Visualization Platform** = React + Three.js frontend (`frontend/`)

---

## Architecture

```
aeris3d/
├── backend/            FastAPI REST API (async job runner)
│   └── main.py
├── core/               All pipeline modules
│   ├── input_manager.py          Phase 1: JPG/PNG/GeoTIFF loading
│   ├── depth_engine.py           Phase 2: DA-V2 / MiDaS / stub
│   ├── structure_engine.py       Phase 3: Classical segmentation
│   ├── hcdc.py                   Phase 4: Hypothesis-Conditioned Depth Correction
│   ├── candidate_generator.py    Phase 5: Multiple height hypotheses
│   ├── counterfactual_height.py  Phase 6+7: CHF + ISCL
│   ├── shadow_test.py            Phase 8: Shadow Contradiction Testing
│   ├── occlusion_test.py         Phase 9: Occlusion-Order Falsification
│   ├── terrain_solver.py         Phase 10: TBCS + scale calibration
│   ├── scoring_engine.py         Phase 11: EGSS
│   ├── self_disproof.py          Phase 12: SDRL
│   ├── uncertainty.py            Phase 13: PDU
│   ├── dsm_exporter.py           Phase 14: ECDSM (PNG/NPZ/GeoTIFF)
│   ├── mesh_builder.py           Phase 15: GLB terrain mesh
│   ├── pipeline.py               Orchestrator (Phases 3–12)
│   ├── config_loader.py          YAML config loading
│   ├── hardware.py               GPU/CPU auto-detection
│   └── logger.py                 Structured logging
├── frontend/           React + Three.js interactive viewer
│   └── src/
│       ├── App.jsx               Main UI (3-panel, AERIS table, charts)
│       ├── ThreeViewer.jsx       3D terrain viewer with flythrough
│       └── index.css             Design system (glassmorphism, badges)
├── configs/
│   └── default.yaml              All tunable parameters
├── data/
│   ├── input/                    User's input images (runtime)
│   ├── cache/                    Depth map cache
│   ├── outputs/                  CLI demo outputs
│   ├── jobs/                     Per-job output (backend)
│   └── dataset/                  GAMUS / training data (separate from runtime)
├── tests/                        62 pytest tests
├── scripts/
│   ├── demo.py                   CLI demo (Gates 1–9)
│   └── train_dataset.py          Dataset inspection + baseline (§8)
├── docs/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11+ (tested on 3.13.5)
- Node.js 18+ (tested on v24.13.0)
- Apple Silicon (MPS) / NVIDIA CUDA / CPU-only

### Python dependencies

```bash
cd ARIES
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `transformers`, `timm`, `fastapi`, `uvicorn`, `rasterio`, `trimesh`, `scikit-image`, `scipy`, `matplotlib`, `pillow`, `pyyaml`

### Frontend dependencies

```bash
cd frontend
npm install
```

---

## Usage

### Demo CLI (Gates 1–9)

```bash
# Run on a real image
python scripts/demo.py --input data/input/sample.jpg

# Run with auto-generated synthetic test image
python scripts/demo.py

# Force CPU
CUDA_VISIBLE_DEVICES="" python scripts/demo.py --input data/input/sample.jpg

# Force re-inference (disable cache)
python scripts/demo.py --input data/input/sample.tif --no-cache
```

**Outputs in `data/outputs/`:**

| File | Description | Gate |
|---|---|---|
| `input_preview.png` | Resized RGB input | 1 |
| `depth.png` | Colorized depth (inferno) | 2 |
| `initial_depth.png` | Depth overlay | 2 |
| `structures.png` | Structure label overlay | 3 |
| `depth_corrected.png` | HCDC-corrected depth | 4 |
| `height_fingerprint.png` | Height vs. consistency chart | 5 |
| `verification_table.png` | AERIS VERIFICATION table image | 6 |
| `candidates.json` | Full candidate table with scores | 5–6 |
| `dsm.png` / `rdsm.png` | Colorized DSM/rDSM | 7 |
| `dsm.npz` | Raw numpy arrays | 7 |
| `uncertainty_report.json` | Per-candidate PDU uncertainty | 8 |
| `mesh.glb` | 3D terrain mesh (vertex-colored) | 9 |
| `results.json` | Full structured pipeline metadata | 6 |

### Backend

```bash
python -m backend.main
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm run dev
# UI: http://localhost:5173
```

### Dataset inspection (§8)

```bash
# Inspect local dataset, generate reports
python scripts/train_dataset.py

# With dataset path override
python scripts/train_dataset.py --dataset-path /path/to/GAMUS

# Skip baseline eval (faster)
python scripts/train_dataset.py --skip-baseline
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/process` | Upload image, start pipeline, return `job_id` |
| GET | `/api/job/{job_id}` | Poll status, progress, results |
| GET | `/api/file/{job_id}/{filename}` | Download output (GLB, PNG, JSON, NPZ) |
| GET | `/api/health` | Health check |

---

## Configuration

All tunable parameters are in [`configs/default.yaml`](configs/default.yaml). No source-code changes needed to switch models, adjust thresholds, or change output paths.

```yaml
depth:
  model: "depth_anything_v2_small"   # or _base, _large, midas_dpt, gradient_stub

candidate_generator:
  min_height_m: 3.0
  max_height_m: 60.0
  step_m: 3.0
  top_k: 5

scoring:
  rejection_threshold: 0.35

mesh:
  resolution: 256
  vertical_exaggeration: 1.5
```

---

## Dataset (§8 — GAMUS)

**Recommended dataset**: GAMUS (`earthflow/GAMUS` on HuggingFace)  
RGB + normalized-DSM tiles across 5 US cities, 6 land-cover classes, pre-split by city (geography-clean).

**Status in this repo**: Dataset not present locally. Run `python scripts/train_dataset.py` to inspect if available, or download via:

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('earthflow/GAMUS')
ds.save_to_disk('data/dataset/raw')
"
```

### §8.5 Comparison Table

| Mode | MAE (scale-inv.) | RMSE (scale-inv.) | Pearson r | Notes |
|---|---|---|---|---|
| A — Pretrained depth only | SKIPPED_NO_REFERENCE_DATA | — | — | Dataset not locally available |
| B — Pretrained depth + AERIS | SKIPPED_NO_REFERENCE_DATA | — | — | Requires dataset for batch eval |
| C — Fine-tuned + AERIS | **SKIPPED** | — | — | Apple M2 MPS insufficient for backbone fine-tuning; explicitly not done |

> **Honest reporting**: Fine-tuning was skipped because the available hardware (Apple M2 MPS, 16 GB) is not a training GPU for ViT-based depth backbones. The pretrained-only baseline is the correct MVP as per §8.4 of the spec. This is documented explicitly here rather than silently omitting the row.

---

## Depth Model Notes

AERIS-3D uses **Depth Anything V2** (HuggingFace) as the primary depth backbone. Downloads automatically on first run.

**Critical domain shift**: DA-V2 and MiDaS are trained primarily on ground-level, driving, and indoor scenes. Nadir (top-down) aerial/satellite imagery represents a significant distribution shift. All depth outputs are treated as **RELATIVE structural cues only**, not metric elevation, and labeled accordingly.

Metric scale anchoring is possible via:
- GeoTIFF pixel size / GSD (when available)
- Reference DEM (CartoDEM / SRTM / Copernicus, when coverage exists)

---

## Evaluation (Official SIH Criteria)

The official evaluation is **50% DSM Estimation Accuracy** + **50% Rendering & UX** (equally weighted).

### DSM Accuracy (50%)
- RMSE, MAE, correlation against reference — computed when reference LiDAR/DEM available
- Stability across landscape types (urban / sparse / hilly / forested)
- When no reference: `"Reference data unavailable — quantitative ground-truth evaluation skipped"`

### Rendering & UX (50%)
- Projection accuracy (source image textured on terrain mesh ✅)
- Visual fidelity (colorized DSM, 3D terrain ✅)
- 3D flythrough navigability (orbit + pan + zoom + auto-rotate ✅)
- Interface intuitiveness (3-panel HUD, AERIS table, evidence panel ✅)
- Software stability (62 tests, no silent failures ✅)
- Standalone deployment (Docker compose provided ✅)

---

## Scientific Honesty

- Depth outputs are labeled **RELATIVE** (not metric) unless a real scale source anchors them.
- Terrain is labeled **RELATIVE** or **ANCHORED_METRIC** — never blurred.
- Quantitative metrics (RMSE, MAE) are only computed when real reference data exists.
- No number displayed in the UI is fabricated — all values come from actual pipeline runs.
- Flood visualization (if enabled) is labeled **"SCENARIO SIMULATION — NOT A FLOOD FORECAST."**
- Fine-tuning: explicitly skipped on M2 MPS with documented reason.

---

## Hardware Support

| Environment | Depth Backbone | Status |
|---|---|---|
| Apple Silicon (MPS) | Depth Anything V2 Small | ✅ Tested |
| NVIDIA CUDA | Depth Anything V2 + mixed precision | ✅ Supported |
| CPU-only | Depth Anything V2 (slow) / gradient_stub | ✅ Fallback |

Auto-detected at startup. No config change needed.

---

## Feature Status

| Feature | Status |
|---|---|
| Input manager (JPG/PNG/GeoTIFF) | ✅ IMPLEMENTED |
| Monocular depth (DA-V2 / MiDaS / stub) | ✅ IMPLEMENTED |
| Hardware auto-detection + MPS/CUDA/CPU | ✅ IMPLEMENTED |
| Depth caching (input_hash + model + config) | ✅ IMPLEMENTED |
| Structural analysis (HCDC) | ✅ IMPLEMENTED |
| Candidate generation (CHF + ISCL) | ✅ IMPLEMENTED |
| Shadow testing (SCT) | ✅ IMPLEMENTED |
| Occlusion testing (OOF) | ✅ IMPLEMENTED |
| Terrain solver (TBCS) | ✅ IMPLEMENTED |
| Evidence scoring (EGSS) | ✅ IMPLEMENTED |
| Self-disproof (SDRL) | ✅ IMPLEMENTED |
| PDU Uncertainty | ✅ IMPLEMENTED |
| DSM/rDSM export (PNG/NPZ/GeoTIFF) | ✅ IMPLEMENTED |
| 3D mesh (GLB, vertex-colored) | ✅ IMPLEMENTED |
| FastAPI backend (async jobs) | ✅ IMPLEMENTED |
| React + Three.js frontend | ✅ IMPLEMENTED |
| AERIS VERIFICATION table (UI) | ✅ IMPLEMENTED |
| Height fingerprint chart (UI) | ✅ IMPLEMENTED |
| Slope assessment display (UI) | ✅ IMPLEMENTED |
| Evidence component bars (UI) | ✅ IMPLEMENTED |
| Uncertainty ± display (UI) | ✅ IMPLEMENTED |
| DSM thumbnail viewer (UI) | ✅ IMPLEMENTED |
| Flythrough controls (orbit/rotate) | ✅ IMPLEMENTED |
| Dataset inspection (§8.1) | ✅ IMPLEMENTED |
| Baseline metrics (§8.3) | ✅ IMPLEMENTED (honest SKIPPED when no data) |
| Depth backbone fine-tuning (§8.4) | ⏭ SKIPPED — M2 MPS not a training GPU |
| Flood simulation | 🔵 PARTIAL — core module exists, UI toggle pending |
| GCP-based scale calibration | 🔵 PARTIAL — architecture exists, requires user-provided GCPs |

---

## Known Limitations

1. **Domain gap**: DA-V2 / MiDaS trained on ground-level imagery — nadir aerial imagery produces relative-only structural cues, not metric elevation.
2. **Fine-tuning**: Not done in this prototype (M2 MPS insufficient for ViT backbone training). Explicitly documented.
3. **Metric calibration**: Requires GeoTIFF CRS or reference DEM. Plain JPG/PNG → RELATIVE output only.
4. **Job persistence**: In-memory job store — jobs lost on backend restart. Sufficient for hackathon demo.
5. **Large images**: Auto-resized to `max_image_size` (default 1024px). Very high-res satellite imagery may lose detail.
6. **Segmentation**: Classical CV (Canny + SLIC) — SAM2 integration is config-available but not the default (compute).

---

## Startup Commands (Standalone Deployment)

```bash
# Terminal 1: Backend
cd ARIES
python -m backend.main
# → http://localhost:8000/api/health

# Terminal 2: Frontend
cd ARIES/frontend
npm run dev
# → http://localhost:5173

# Or with Docker:
docker-compose up
```

---

## Running Tests

```bash
cd ARIES
python -m pytest tests/ -v
# Expected: 62 passed
```

---

## Demo Command

```bash
python scripts/demo.py --input data/input/synthetic_test.jpg
# Expected output: Gates 1–9 all ✓, mesh.glb produced
```
