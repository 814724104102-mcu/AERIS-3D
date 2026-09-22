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

