"""
AERIS-3D — Interactive Web Application & API Service
Provides REST endpoints and a full Three.js 3D flythrough viewer for
single-view height estimation and 3D terrain reconstruction.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import numpy as np
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config_loader import config_version_hash, load_config
from core.depth_engine import DepthEngine
from core.hardware import get_hardware
from core.input_manager import load_input
from core.logger import get_logger
from core.mesh_builder import build_mesh
from core.pipeline import run_pipeline

log = get_logger("webapp")

app = FastAPI(
    title="AERIS-3D — 3D Reconstruction Platform",
    description="Smart India Hackathon 2026 — Problem SIH26175 — DepthWizard (ISRO)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()
cfg_hash = config_version_hash(cfg)
hw = get_hardware()

if not hw.torch_available:
    cfg.setdefault("depth", {})["model"] = "gradient_stub"

depth_engine = DepthEngine(cfg)

JOBS_DIR = _REPO_ROOT / "data" / "jobs"
OUTPUTS_DIR = _REPO_ROOT / "data" / "outputs"
INPUT_DIR = _REPO_ROOT / "data" / "input"


@app.get("/api/health")
def api_health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/hardware")
def api_hardware():
    return {
        "device": hw.device,
        "device_name": hw.device_name,
        "cuda_available": hw.cuda_available,
        "mps_available": hw.mps_available,
        "torch_available": hw.torch_available,
        "cpu_count": hw.cpu_count,
        "model": cfg.get("depth", {}).get("model", "gradient_stub"),
    }


@app.get("/api/jobs")
def api_list_jobs():
    jobs = []
    
    demo_mesh = OUTPUTS_DIR / "demo_run" / "mesh.glb"
    if demo_mesh.exists():
        jobs.append({
            "id": "demo_run",
            "name": "Live Demo Run (Synthetic Test)",
            "type": "live",
            "has_mesh": True,
            "has_depth": (OUTPUTS_DIR / "demo_run" / "depth_preview.png").exists(),
            "has_input": (OUTPUTS_DIR / "demo_run" / "input_preview.png").exists(),
        })

    if JOBS_DIR.exists():
        for job_folder in sorted(JOBS_DIR.iterdir(), reverse=True):
            if job_folder.is_dir():
                results_file = job_folder / "results.json"
                mesh_file = job_folder / "mesh.glb"
                name = f"Job {job_folder.name[:8]}"
                metadata = {}
                if results_file.exists():
                    try:
                        with open(results_file, "r") as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                jobs.append({
                    "id": job_folder.name,
                    "name": name,
                    "type": "benchmark",
                    "has_mesh": mesh_file.exists(),
                    "has_depth": (job_folder / "depth.png").exists(),
                    "has_input": (job_folder / "input.jpg").exists(),
                    "metadata": metadata,
                })

    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    if job_id == "demo_run":
        job_path = OUTPUTS_DIR / "demo_run"
    else:
        job_path = JOBS_DIR / job_id

    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    results_file = job_path / "results.json"
    data = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
    else:
        data = {
            "job_id": job_id,
            "status": "COMPLETED",
            "mesh_available": (job_path / "mesh.glb").exists(),
        }

    files = [f.name for f in job_path.iterdir() if f.is_file()]
    data["available_files"] = files
    data["status"] = "done"
    return data


@app.get("/api/job/{job_id}")
def api_get_job_alias(job_id: str):
    return api_get_job(job_id)


@app.get("/api/jobs/{job_id}/files/{filename}")
def api_get_job_file(job_id: str, filename: str):
    if job_id == "demo_run":
        file_path = OUTPUTS_DIR / "demo_run" / filename
    else:
        file_path = JOBS_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    media_type = "application/octet-stream"
    if filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".json"):
        media_type = "application/json"

    return FileResponse(file_path, media_type=media_type)


@app.get("/api/file/{job_id}/{filename}")
def api_get_file_alias(job_id: str, filename: str):
    return api_get_job_file(job_id, filename)


@app.get("/api/samples")
def api_list_samples():
    samples = []
    for f in INPUT_DIR.glob("*.*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                with PILImage.open(f) as img:
                    samples.append({
                        "name": f.name,
                        "size": f"{img.width}x{img.height}",
                        "format": img.format
                    })
            except Exception:
                pass
    return {"samples": samples}


def _process_image_job(input_img_path: Path, job_dir: Path, resolution: int = 200, v_exag: float = 1.5):
    t0 = time.perf_counter()
    input_data = load_input(input_img_path, cfg)
    depth_res = depth_engine.estimate(input_data.image_rgb, input_data.input_hash, cfg_hash)

    depth_img = (depth_res.normalized_depth * 255).astype(np.uint8)
    PILImage.fromarray(depth_img).save(job_dir / "depth.png")
    PILImage.fromarray(input_data.image_rgb).save(job_dir / "input_preview.png")

    custom_cfg = dict(cfg)
    custom_cfg.setdefault("mesh", {})["resolution"] = resolution
    custom_cfg["mesh"]["vertical_exaggeration"] = v_exag

    mesh_res = build_mesh(
        depth_res.normalized_depth,
        input_data.image_rgb,
        "RELATIVE",
        job_dir,
        custom_cfg
    )

    pipeline_res = run_pipeline(input_data, depth_res, cfg)

    total_time = time.perf_counter() - t0

    summary = {
        "job_id": job_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_file": input_img_path.name,
        "image_size": f"{input_data.preprocessing_meta.original_width}x{input_data.preprocessing_meta.original_height}",
        "mesh": {
            "vertices": mesh_res.vertex_count,
            "faces": mesh_res.face_count,
            "vertical_exaggeration": v_exag,
            "resolution": resolution,
            "glb_file": "mesh.glb"
        },
        "depth": {
            "model": depth_res.depth_metadata.model_name,
            "relative_only": depth_res.depth_metadata.relative_only,
            "runtime_s": round(depth_res.depth_metadata.runtime_s, 3)
        },
        "pipeline": {
            "candidates_evaluated": pipeline_res.n_candidates,
            "survived": pipeline_res.n_survived,
            "rejected": pipeline_res.n_rejected,
        },
        "total_runtime_s": round(total_time, 3),
        "status": "done"
    }

    with open(job_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


@app.post("/api/process")
async def api_process(
    file: Optional[UploadFile] = File(None),
):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_img_path = job_dir / "input.jpg"
    if file and file.filename:
        content = await file.read()
        with open(input_img_path, "wb") as f:
            f.write(content)
    else:
        src = INPUT_DIR / "synthetic_test.jpg"
        shutil.copyfile(src, input_img_path)

    summary = _process_image_job(input_img_path, job_dir)
    return {"job_id": job_id, "status": "done", "summary": summary}


@app.post("/api/reconstruct")
async def api_reconstruct(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    resolution: int = Form(200),
    v_exag: float = Form(1.5),
):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_img_path = job_dir / "input.jpg"

    if file and file.filename:
        content = await file.read()
        with open(input_img_path, "wb") as f:
            f.write(content)
    elif sample_name:
        src = INPUT_DIR / sample_name
        if not src.exists():
            raise HTTPException(status_code=400, detail="Sample image not found")
        shutil.copyfile(src, input_img_path)
    else:
        src = INPUT_DIR / "synthetic_test.jpg"
        shutil.copyfile(src, input_img_path)

    summary = _process_image_job(input_img_path, job_dir, resolution, v_exag)
    return summary

HTML_PAGE = """<!DOCTYPE html><!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AERIS-3D | ISRO 3D Flythrough & Elevation Platform</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/PointerLockControls.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0d14;
            --bg-secondary: #111827;
            --card-bg: rgba(17, 24, 39, 0.85);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-blue: #38bdf8;
            --accent-orange: #fb923c;
            --accent-green: #34d399;
            --text-main: #f3f4f6;
            --text-dim: #9ca3af;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Grotesk', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            height: 60px;
            background: rgba(10, 13, 20, 0.95);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            backdrop-filter: blur(12px);
            z-index: 50;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .badge-sih {
            background: linear-gradient(135deg, #ea580c, #c2410c);
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }
        .title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .title span { color: var(--accent-blue); }

        .system-status {
            display: flex;
            align-items: center;
            gap: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
        }

        .workspace {
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }

        #viewer-container {
            flex: 1;
            height: 100%;
            position: relative;
            background: radial-gradient(circle at center, #172033 0%, #080b12 100%);
        }
        #three-canvas { width: 100%; height: 100%; display: block; }

        .viewer-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 10px 14px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            display: flex;
            gap: 10px;
            z-index: 10;
        }
        .btn-ctrl {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'Space Grotesk', sans-serif;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-ctrl:hover, .btn-ctrl.active {
            background: var(--accent-blue);
            color: #000;
            font-weight: 600;
        }

        .sidebar {
            width: 440px;
            background: var(--card-bg);
            border-left: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            z-index: 20;
        }
        .sidebar-tabs {
            display: flex;
            border-bottom: 1px solid var(--card-border);
            background: rgba(0, 0, 0, 0.2);
        }
        .tab-btn {
            flex: 1;
            padding: 12px 8px;
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn.active {
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
            background: rgba(56, 189, 248, 0.05);
        }

        .tab-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .panel-section {
            margin-bottom: 20px;
        }
        .panel-section h3 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 12px;
        }

        select, input[type="file"], input[type="range"] {
            width: 100%;
            padding: 10px 12px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            color: var(--text-main);
            font-family: inherit;
            margin-bottom: 12px;
            font-size: 13px;
        }
        select:focus { outline: 1px solid var(--accent-blue); }

        .btn-primary {
            width: 100%;
            background: linear-gradient(135deg, #0284c7, #0369a1);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .btn-primary:hover { background: linear-gradient(135deg, #38bdf8, #0284c7); color: #000; }

        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 16px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px 12px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 16px;
            font-weight: 700;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
            margin-top: 4px;
        }

        .preview-box {
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--card-border);
            background: #000;
            margin-bottom: 12px;
        }
        .preview-box img {
            width: 100%;
            display: block;
            object-fit: contain;
            max-height: 240px;
        }

        .spinner {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            z-index: 100;
            background: rgba(10, 13, 20, 0.85);
            padding: 24px 36px;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            backdrop-filter: blur(8px);
        }
        .spin-circle {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(56, 189, 248, 0.2);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        pre {
            background: rgba(0, 0, 0, 0.5);
            padding: 12px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #a5f3fc;
            overflow-x: auto;
            max-height: 350px;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="badge-sih">SIH 2026 • ISRO</div>
            <div class="title">AERIS-3D <span>Explorer</span></div>
        </div>
        <div class="system-status">
            <div class="status-dot"></div>
            <span id="hw-status">CPU Mode • Active</span>
        </div>
    </header>

    <div class="workspace">
        <div id="viewer-container">
            <div class="viewer-overlay">
                <button class="btn-ctrl" id="btn-fly" onclick="toggleFlythrough()">Auto-Flythrough</button>
                <button class="btn-ctrl" id="btn-wire" onclick="toggleWireframe()">Wireframe</button>
                <button class="btn-ctrl" onclick="resetCamera()">Reset Camera</button>
                <button class="btn-ctrl" id="btn-pointerlock" onclick="startPointerLock()">Fly Mode (PointerLock)</button>
                <button class="btn-ctrl" id="btn-slope" onclick="toggleSlopeTool()">Slope Tool</button>
                <button class="btn-ctrl" id="btn-overlay" onclick="toggleReferenceOverlay()">Ref Overlay</button>
                <div id="height-readout" style="color: #34d399; font-family: monospace; display: flex; align-items: center; margin-left: 10px;">Height: --</div>
                <div id="slope-readout" style="color: #fb923c; font-family: monospace; display: flex; align-items: center; margin-left: 10px;">Slope: --</div>

            </div>
            <div id="loading-indicator" class="spinner">
                <div class="spin-circle"></div>
                <div id="loading-text" style="font-size: 13px; font-weight: 500;">Reconstructing 3D Mesh...</div>
            </div>
            <canvas id="three-canvas"></canvas>
        </div>

        <div class="sidebar">
            <div class="sidebar-tabs">
                <button class="tab-btn active" onclick="switchTab('reconstruct')">Reconstruct</button>
                <button class="tab-btn" onclick="switchTab('analysis')">Analysis</button>
                <button class="tab-btn" onclick="switchTab('metadata')">Metadata</button>
            </div>

            <div id="tab-reconstruct" class="tab-content">
                <div class="panel-section">
                    <h3>Select Pre-Computed Job</h3>
                    <select id="job-select" onchange="loadSelectedJob()">
                        <option value="">Loading jobs...</option>
                    </select>
                </div>

                <div class="panel-section">
                    <h3>Run New 3D Estimation</h3>
                    <label style="font-size: 12px; color: var(--text-dim); display:block; margin-bottom: 6px;">Sample Remote Sensing Imagery</label>
                    <select id="sample-select">
                        <option value="synthetic_test.jpg">synthetic_test.jpg (512x512)</option>
                        <option value="test.jpg">test.jpg (100x100)</option>
                        <option value="test.png">test.png (100x100)</option>
                    </select>

                    <label style="font-size: 12px; color: var(--text-dim); display:block; margin-bottom: 6px;">Or Upload Satellite / Aerial Image (JPG/PNG)</label>
                    <input type="file" id="image-upload" accept="image/*">

                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 12px; color: var(--text-dim);">Vertical Exaggeration</span>
                        <span id="v-exag-val" style="font-size: 12px; font-family: monospace; color: var(--accent-blue);">1.5x</span>
                    </div>
                    <input type="range" id="v-exag" min="0.5" max="3.0" step="0.1" value="1.5" oninput="document.getElementById('v-exag-val').textContent = this.value + 'x'">

                    <button class="btn-primary" onclick="runReconstruction()">
                        Run 3D Reconstruction
                    </button>
                </div>

                <div class="panel-section">
                    <h3>Active Model Details</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Vertices</div>
                            <div class="metric-value" id="val-verts">--</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Triangles</div>
                            <div class="metric-value" id="val-faces">--</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Scale Mode</div>
                            <div class="metric-value" id="val-scale" style="font-size: 12px;">RELATIVE</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Inference</div>
                            <div class="metric-value" id="val-time">--</div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="tab-analysis" class="tab-content" style="display: none;">
                <div class="panel-section">
                    <h3>Input Aerial Image</h3>
                    <div class="preview-box">
                        <img id="img-input" src="" alt="Input preview">
                    </div>
                </div>

                <div class="panel-section">
                    <h3>Estimated Surface / Depth Map</h3>
                    <div class="preview-box">
                        <img id="img-depth" src="" alt="Depth map">
                    </div>
                </div>

                <div class="panel-section" id="section-fingerprint" style="display: none;">
                    <h3>Height Fingerprint (CHF)</h3>
                    <div class="preview-box">
                        <img id="img-fingerprint" src="" alt="Height fingerprint">
                    </div>
                </div>

                <div class="panel-section" id="section-verification" style="display: none;">
                    <h3>AERIS Verification Table</h3>
                    <div class="preview-box">
                        <img id="img-table" src="" alt="Verification table">
                    </div>
                </div>
            </div>

            <div id="tab-metadata" class="tab-content" style="display: none;">
                <div class="panel-section">
                    <h3>Job Metadata & EGSS Scores</h3>
                    <pre id="json-metadata">Select or run a job to inspect metadata.</pre>
                </div>
            </div>
        </div>
    </div>

    <script>
        let scene, camera, renderer, controls, currentMesh = null;
        let isFlythrough = false;
        let isWireframe = false;

        let pointerLockControls;
        let isPointerLocked = false;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2(0, 0); // Center of screen for pointer lock, or mouse pos
        let slopeMode = false;
        let slopePoints = [];
        let slopeMarkers = [];
        let referenceMode = false;
        let originalMaterial = null;
        let referenceMaterial = new THREE.MeshBasicMaterial({color: 0xff0000, wireframe: true}); // Mock for reference

        let currentJobId = null;

        function initViewer() {
            const container = document.getElementById('viewer-container');
            const canvas = document.getElementById('three-canvas');

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0f1d);

            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 80, 120);

            renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.toneMapping = THREE.ACESFilmicToneMapping;

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 - 0.05;

            pointerLockControls = new THREE.PointerLockControls(camera, document.body);
            pointerLockControls.addEventListener('lock', () => { isPointerLocked = true; });
            pointerLockControls.addEventListener('unlock', () => { isPointerLocked = false; });
            
            document.addEventListener('click', onDocumentClick, false);


            const ambient = new THREE.AmbientLight(0xffffff, 0.8);
            scene.add(ambient);

            const sun = new THREE.DirectionalLight(0xffffff, 1.2);
            sun.position.set(100, 150, 100);
            scene.add(sun);

            const fill = new THREE.DirectionalLight(0x38bdf8, 0.5);
            fill.position.set(-100, 50, -100);
            scene.add(fill);

            const grid = new THREE.GridHelper(200, 40, 0x1e293b, 0x0f172a);
            grid.position.y = -2;
            scene.add(grid);

            window.addEventListener('resize', onWindowResize);
            animate();
        }

        function onWindowResize() {
            const container = document.getElementById('viewer-container');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        let angle = 0;
        
        function animate() {
            requestAnimationFrame(animate);

            if (isFlythrough) {
                angle += 0.005;
                const radius = 90;
                camera.position.x = Math.sin(angle) * radius;
                camera.position.z = Math.cos(angle) * radius;
                camera.position.y = 45 + Math.sin(angle * 2) * 10;
                controls.target.set(0, 5, 0);
            }

            // Height readout
            if (currentMesh) {
                raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
                const intersects = raycaster.intersectObject(currentMesh, true);
                if (intersects.length > 0) {
                    const h = intersects[0].point.y;
                    document.getElementById('height-readout').textContent = `Height: ${h.toFixed(2)}m`;
                } else {
                    document.getElementById('height-readout').textContent = `Height: --`;
                }
                
                // Collision / Clamping
                if (isPointerLocked || isFlythrough || true) {
                    // Raycast down from camera X,Z to find terrain height
                    const downRay = new THREE.Raycaster(new THREE.Vector3(camera.position.x, 1000, camera.position.z), new THREE.Vector3(0, -1, 0));
                    const downHits = downRay.intersectObject(currentMesh, true);
                    if (downHits.length > 0) {
                        const groundY = downHits[0].point.y;
                        if (camera.position.y < groundY + 2) {
                            camera.position.y = groundY + 2;
                        }
                    }
                }
            }

            if (!isPointerLocked) controls.update();
            renderer.render(scene, camera);
        }


        
        function startPointerLock() {
            pointerLockControls.lock();
        }

        function toggleSlopeTool() {
            slopeMode = !slopeMode;
            document.getElementById('btn-slope').classList.toggle('active', slopeMode);
            slopePoints = [];
            slopeMarkers.forEach(m => scene.remove(m));
            slopeMarkers = [];
            document.getElementById('slope-readout').textContent = 'Slope: --';
        }

        function toggleReferenceOverlay() {
            referenceMode = !referenceMode;
            document.getElementById('btn-overlay').classList.toggle('active', referenceMode);
            if (currentMesh) {
                currentMesh.traverse((child) => {
                    if (child.isMesh) {
                        if (referenceMode) {
                            if (!originalMaterial) originalMaterial = child.material;
                            child.material = referenceMaterial;
                        } else {
                            if (originalMaterial) child.material = originalMaterial;
                        }
                    }
                });
            }
        }

        function onDocumentClick(event) {
            if (!slopeMode || !currentMesh) return;
            
            // If pointer locked, use center. Else use mouse coordinates
            if (!isPointerLocked) {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            } else {
                mouse.x = 0; mouse.y = 0;
            }

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(currentMesh, true);
            
            if (intersects.length > 0) {
                const pt = intersects[0].point;
                slopePoints.push(pt);
                
                const geo = new THREE.SphereGeometry(1, 16, 16);
                const mat = new THREE.MeshBasicMaterial({color: 0xfb923c});
                const sphere = new THREE.Mesh(geo, mat);
                sphere.position.copy(pt);
                scene.add(sphere);
                slopeMarkers.push(sphere);

                if (slopePoints.length === 2) {
                    const p1 = slopePoints[0];
                    const p2 = slopePoints[1];
                    const dy = Math.abs(p2.y - p1.y);
                    const dxz = Math.hypot(p2.x - p1.x, p2.z - p1.z);
                    const angleRad = Math.atan2(dy, dxz);
                    const angleDeg = (angleRad * 180 / Math.PI).toFixed(1);
                    const grade = ((dy / dxz) * 100).toFixed(1);
                    document.getElementById('slope-readout').textContent = `Slope: ${angleDeg}° (${grade}%)`;
                    
                    // reset for next pair
                    slopePoints = [];
                    setTimeout(() => {
                        slopeMarkers.forEach(m => scene.remove(m));
                        slopeMarkers = [];
                    }, 3000);
                }
            }
        }

        // We will override animate to add our features
function toggleFlythrough() {
            isFlythrough = !isFlythrough;
            document.getElementById('btn-fly').classList.toggle('active', isFlythrough);
        }

        function toggleWireframe() {
            isWireframe = !isWireframe;
            document.getElementById('btn-wire').classList.toggle('active', isWireframe);
            if (currentMesh) {
                currentMesh.traverse((child) => {
                    if (child.isMesh) {
                        child.material.wireframe = isWireframe;
                    }
                });
            }
        }

        function resetCamera() {
            camera.position.set(0, 80, 120);
            controls.target.set(0, 0, 0);
            isFlythrough = false;
            document.getElementById('btn-fly').classList.remove('active');
        }

        function loadGLB(url) {
            showLoading(true, "Loading 3D Terrain Model...");
            const loader = new THREE.GLTFLoader();

            loader.load(url, (gltf) => {
                if (currentMesh) {
                    scene.remove(currentMesh);
                }

                currentMesh = gltf.scene;

                const box = new THREE.Box3().setFromObject(currentMesh);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                currentMesh.position.x -= center.x;
                currentMesh.position.y -= center.y;
                currentMesh.position.z -= center.z;

                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 80 / (maxDim || 1);
                currentMesh.scale.set(scale, scale, scale);

                let vertCount = 0;
                let faceCount = 0;

                currentMesh.traverse((child) => {
                    if (child.isMesh) {
                        vertCount += child.geometry.attributes.position.count;
                        if (child.geometry.index) {
                            faceCount += child.geometry.index.count / 3;
                        }
                        child.material.wireframe = isWireframe;
                        child.material.side = THREE.DoubleSide;
                    }
                });

                document.getElementById('val-verts').textContent = vertCount.toLocaleString();
                document.getElementById('val-faces').textContent = Math.round(faceCount).toLocaleString();

                scene.add(currentMesh);
                showLoading(false);
            }, undefined, (err) => {
                console.error("Error loading GLB:", err);
                showLoading(false);
            });
        }

        function showLoading(show, text="Processing...") {
            const el = document.getElementById('loading-indicator');
            el.style.display = show ? 'flex' : 'none';
            document.getElementById('loading-text').textContent = text;
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');

            if (tabId === 'reconstruct') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('tab-reconstruct').style.display = 'block';
            } else if (tabId === 'analysis') {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('tab-analysis').style.display = 'block';
            } else if (tabId === 'metadata') {
                document.querySelectorAll('.tab-btn')[2].classList.add('active');
                document.getElementById('tab-metadata').style.display = 'block';
            }
        }

        async function fetchJobs() {
            const res = await fetch('/api/jobs');
            const data = await res.json();
            const select = document.getElementById('job-select');
            select.innerHTML = '';

            data.jobs.forEach(j => {
                const opt = document.createElement('option');
                opt.value = j.id;
                opt.textContent = `${j.name} (${j.type})`;
                select.appendChild(opt);
            });

            if (data.jobs.length > 0) {
                select.value = data.jobs[0].id;
                loadSelectedJob();
            }
        }

        async function loadSelectedJob() {
            const jobId = document.getElementById('job-select').value;
            if (!jobId) return;
            currentJobId = jobId;

            const res = await fetch(`/api/jobs/${jobId}`);
            const meta = await res.json();

            document.getElementById('json-metadata').textContent = JSON.stringify(meta, null, 2);

            const imgInput = document.getElementById('img-input');
            const imgDepth = document.getElementById('img-depth');

            if (jobId === 'demo_run') {
                imgInput.src = `/api/jobs/${jobId}/files/input_preview.png`;
                imgDepth.src = `/api/jobs/${jobId}/files/depth_preview.png`;
            } else {
                imgInput.src = `/api/jobs/${jobId}/files/input.jpg`;
                imgDepth.src = `/api/jobs/${jobId}/files/depth.png`;
            }

            const secFP = document.getElementById('section-fingerprint');
            const secVT = document.getElementById('section-verification');
            if (meta.available_files && meta.available_files.includes('height_fingerprint.png')) {
                secFP.style.display = 'block';
                document.getElementById('img-fingerprint').src = `/api/jobs/${jobId}/files/height_fingerprint.png`;
            } else {
                secFP.style.display = 'none';
            }

            if (meta.available_files && meta.available_files.includes('verification_table.png')) {
                secVT.style.display = 'block';
                document.getElementById('img-table').src = `/api/jobs/${jobId}/files/verification_table.png`;
            } else {
                secVT.style.display = 'none';
            }

            loadGLB(`/api/jobs/${jobId}/files/mesh.glb`);
        }

        async function runReconstruction() {
            const fileInput = document.getElementById('image-upload');
            const sampleSelect = document.getElementById('sample-select');
            const vExag = document.getElementById('v-exag').value;

            const formData = new FormData();
            formData.append('resolution', 200);
            formData.append('v_exag', vExag);

            if (fileInput.files.length > 0) {
                formData.append('file', fileInput.files[0]);
            } else {
                formData.append('sample_name', sampleSelect.value);
            }

            showLoading(true, "Inferring Depth & Building 3D Mesh...");
            const t0 = performance.now();

            try {
                const res = await fetch('/api/reconstruct', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
                document.getElementById('val-time').textContent = `${elapsed}s`;

                await fetchJobs();
                document.getElementById('job-select').value = data.job_id;
                await loadSelectedJob();
            } catch (err) {
                alert("Reconstruction error: " + err);
                showLoading(false);
            }
        }

        async function checkHardware() {
            try {
                const res = await fetch('/api/hardware');
                const hw = await res.json();
                document.getElementById('hw-status').textContent = 
                    `${hw.device.toUpperCase()} (${hw.device_name || 'CPU'}) • Model: ${hw.model}`;
            } catch (e) {}
        }

        window.onload = () => {
            initViewer();
            checkHardware();
            fetchJobs();
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
