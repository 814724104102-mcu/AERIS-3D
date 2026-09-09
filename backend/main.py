"""
AERIS-3D — FastAPI Backend (Phase 17)

Endpoints:
  POST /api/process         — upload image, run pipeline, return job_id
  GET  /api/job/{job_id}    — get job status + full results JSON
  GET  /api/file/{job_id}/{filename} — serve output file (GLB, PNG, JSON, NPZ)
  GET  /api/health          — health check

Jobs run in a background ThreadPoolExecutor (avoids blocking the event loop).
Results are stored in an in-memory dict (sufficient for hackathon).

All CORS origins from config are allowed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.config_loader import config_version_hash, load_config
from core.logger import get_logger

log = get_logger("backend")

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AERIS-3D API",
    version="0.2.0",
    description="AERIS-3D — Hypothesis-Testing 3D Reconstruction from Monocular Imagery",
)

cfg = load_config()
cfg_hash = config_version_hash(cfg)

CORS_ORIGINS = cfg.get("backend", {}).get(
    "cors_origins",
    ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev mode — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job store: {job_id: {status, result, error, output_dir}}
jobs: dict[str, dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=2)

JOBS_DIR = Path("data/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload an image and start the AERIS-3D pipeline.
    Returns a job_id immediately. Poll /api/job/{job_id} for status.
    """
    job_id = str(uuid.uuid4())[:12]
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    img_ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    input_path = output_dir / f"input{img_ext}"
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    log.info("Job %s: received %s (%d bytes)", job_id, file.filename, len(content))

    # Initialize job record
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "result": None,
        "error": None,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
    }

    # Launch pipeline in thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_pipeline_job, job_id, input_path, output_dir)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs[job_id]
    return job


@app.get("/api/file/{job_id}/{filename:path}")
async def get_output_file(job_id: str, filename: str):
    """Serve an output file for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(jobs[job_id]["output_dir"])
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        ".glb": "model/gltf-binary",
        ".png": "image/png",
        ".json": "application/json",
        ".npz": "application/octet-stream",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (for debugging)."""
    return [
        {
            "job_id": jid,
            "status": info["status"],
            "stage": info.get("stage"),
            "progress": info.get("progress"),
        }
        for jid, info in jobs.items()
    ]


# ─────────────────────────────────────────────────────────────
# Background pipeline runner
# ─────────────────────────────────────────────────────────────


def _progress(job_id: str, stage: str, pct: int) -> None:
    if job_id in jobs:
        jobs[job_id]["stage"] = stage
        jobs[job_id]["progress"] = pct
        log.info("Job %s [%d%%] %s", job_id, pct, stage)


def _run_pipeline_job(job_id: str, input_path: Path, output_dir: Path) -> None:
    """
    Full AERIS-3D pipeline run for one uploaded image.
    Updates jobs[job_id] throughout.
    """
    try:
        jobs[job_id]["status"] = "running"
        _progress(job_id, "Loading input", 5)

        from core.input_manager import load_input

        input_data = load_input(input_path, cfg)
        _progress(job_id, "Depth inference", 15)

        from core.depth_engine import DepthEngine

        engine = DepthEngine(cfg)
        depth_result = engine.estimate(
            input_data.image_rgb, input_data.input_hash, cfg_hash
        )
        _progress(job_id, "Structure analysis", 35)

        from core.pipeline import run_pipeline

        pipeline_result = run_pipeline(input_data, depth_result, cfg)
        pr = pipeline_result
        _progress(job_id, "Uncertainty analysis", 55)

        from core.uncertainty import run_pdu

        pdu_result = run_pdu(
            pr.candidates, pr.evidence_scores, pr.hcdc_result.corrected_depth, cfg
        )
        _progress(job_id, "Exporting DSM", 65)

        from core.dsm_exporter import export_dsm

        dsm_result = export_dsm(
            pr.hcdc_result.corrected_depth,
            pr.terrain_result,
            input_data.georef,
            output_dir,
            cfg,
        )
        _progress(job_id, "Building mesh", 78)

        from core.mesh_builder import build_mesh

        mesh_result = build_mesh(
            dsm_result.dsm,
            input_data.image_rgb,
            pr.terrain_result.scale_mode,
            output_dir,
            cfg,
        )
        _progress(job_id, "Generating visualizations", 88)

        # Save standard output images
        _save_outputs(pr, output_dir)
        _progress(job_id, "Saving results", 95)

        # Build results JSON
        sdrl = pr.sdrl_result
        result_data = {
            "aeris3d_version": "0.2.0",
            "job_id": job_id,
            "input": {
                "file": input_path.name,
                "format": input_data.file_format.value,
                "size": {
                    "height": input_data.preprocessing_meta.original_height,
                    "width": input_data.preprocessing_meta.original_width,
                },
                "georef_present": input_data.georef is not None,
            },
            "depth": {
                "model": depth_result.depth_metadata.model_name,
                "device": depth_result.depth_metadata.device,
                "relative_only": depth_result.depth_metadata.relative_only,
                "runtime_s": round(depth_result.depth_metadata.runtime_s, 3),
            },
            "structure": {
                "num_regions": pr.structure_result.num_regions,
                "num_buildings": len(pr.structure_result.buildings),
                "segmentation_quality": round(
                    pr.structure_result.segmentation_quality, 3
                ),
                "backend": pr.structure_result.backend_used,
            },
            "terrain": {
                "scale_mode": pr.terrain_result.scale_mode,
                "terrain_depth_level": round(pr.terrain_result.terrain_depth_level, 4),
            },
            "candidates": {
                "total": pr.n_candidates,
                "survived": pr.n_survived,
                "rejected": pr.n_rejected,
            },
            "shadow": {
                "available": pr.shadow_result.shadow_available,
                "reliable": pr.shadow_result.shadow_reliable,
                "effective_weight": round(pr.shadow_result.effective_weight, 3),
            },
            "survivors": {
                obj_id: {
                    "candidate_id": es.candidate_id,
                    "height_value": round(es.height_value, 2),
                    "height_unit": es.height_unit,
                    "overall_score": round(es.overall_score, 4),
                    "component_scores": {
                        k: round(v, 4) for k, v in es.component_scores.items()
                    },
                    "active_evidence": es.active_evidence,
                    "weights": {k: round(v, 4) for k, v in es.weights.items()},
                }
                for obj_id, es in sdrl.survivors.items()
            },
            "candidates_table": [
                {
                    "candidate_id": cand.candidate_id,
                    "object_id": cand.object_id,
                    "height_value": round(cand.height_value, 2),
                    "height_unit": cand.height_unit,
                    "status": (
                        "SURVIVED"
                        if cand.candidate_id
                        in {v.candidate_id for v in sdrl.survivors.values()}
                        else "REJECTED"
                    ),
                    "overall_score": round(
                        next(
                            (
                                e.overall_score
                                for e in pr.evidence_scores
                                if e.candidate_id == cand.candidate_id
                            ),
                            0,
                        ),
                        4,
                    ),
                    "component_scores": next(
                        (
                            {k: round(v, 4) for k, v in e.component_scores.items()}
                            for e in pr.evidence_scores
                            if e.candidate_id == cand.candidate_id
                        ),
                        {},
                    ),
                }
                for cand in pr.candidates
            ],
            "uncertainty": [
                {
                    "candidate_id": u.candidate_id,
                    "object_id": u.object_id,
                    "height_value": round(u.height_value, 2),
                    "nominal_score": round(u.nominal_score, 4),
                    "mean_score": round(u.mean_score, 4),
                    "std_score": round(u.std_score, 4),
                    "p05": round(u.p05, 4),
                    "p95": round(u.p95, 4),
                    "confidence_label": u.confidence_label,
                }
                for u in pdu_result.uncertainties
            ],
            "height_fingerprint": pr.chf_result.height_fingerprint,
            "sdrl_log": pr.sdrl_result.all_iterations_table[:30],
            "dsm": {
                "scale_mode": dsm_result.scale_mode,
                "exported_files": dsm_result.exported_files,
                "stats": dsm_result.stats,
            },
            "mesh": {
                "vertex_count": mesh_result.vertex_count,
                "face_count": mesh_result.face_count,
                "vertical_exaggeration": mesh_result.vertical_exaggeration,
                "warning": mesh_result.warning,
            },
            "phase_runtimes_s": {k: round(v, 3) for k, v in pr.phase_runtimes.items()},
            "total_runtime_s": round(pr.total_runtime_s, 3),
            "output_files": [
                "input_preview.png",
                "depth.png",
                "structures.png",
                "depth_corrected.png",
                "height_fingerprint.png",
                "verification_table.png",
                "dsm.png",
                "rdsm.png",
                "dsm.npz",
                "dsm_metadata.json",
                "mesh.glb",
            ],
        }

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result_data
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Complete"
        log.info("Job %s: DONE in %.1fs", job_id, pr.total_runtime_s)

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Job %s FAILED: %s\n%s", job_id, exc, tb)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["stage"] = "Error"


def _save_outputs(pr, output_dir: Path) -> None:
    """Save standard visualization outputs to output_dir."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import sys
    from pathlib import Path as P

    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    demo_path = _REPO_ROOT / "scripts" / "demo.py"
    # Import helpers from demo without running __main__
    spec_source = open(demo_path).read()
    ns: dict = {"__file__": str(demo_path)}
    exec(compile(spec_source.split("if __name__")[0], str(demo_path), "exec"), ns)

    colorize_depth = ns.get("colorize_depth")
    make_structure_overlay = ns.get("make_structure_overlay")
    make_height_fingerprint_chart = ns.get("make_height_fingerprint_chart")
    make_candidate_table_image = ns.get("make_candidate_table_image")
    save_image = ns.get("save_image")

    image_rgb = pr.input_data.image_rgb
    depth_result = pr.depth_result
    nd = depth_result.normalized_depth

    # Input preview
    pil_prev = PILImage.fromarray(image_rgb)
    if max(pil_prev.size) > 1024:
        pil_prev.thumbnail((1024, 1024), PILImage.LANCZOS)
    pil_prev.save(output_dir / "input_preview.png")

    # Depth
    save_image(colorize_depth(nd), output_dir / "depth.png")

    # Depth overlay
    pil_rgb = PILImage.fromarray(image_rgb)
    pil_dep = PILImage.fromarray(colorize_depth(nd))
    pil_dep = pil_dep.resize(pil_rgb.size, PILImage.BILINEAR)
    PILImage.blend(pil_rgb, pil_dep, 0.5).save(output_dir / "initial_depth.png")

    # Structures
    save_image(
        make_structure_overlay(image_rgb, pr.structure_result),
        output_dir / "structures.png",
    )

    # Corrected depth
    corr_nd = pr.hcdc_result.corrected_depth
    corr_min, corr_max = corr_nd.min(), corr_nd.max()
    corr_norm = ((corr_nd - corr_min) / max(corr_max - corr_min, 1e-6)).astype(
        np.float32
    )
    save_image(colorize_depth(corr_norm), output_dir / "depth_corrected.png")

    # Charts
    make_height_fingerprint_chart(
        pr.chf_result, pr.sdrl_result, output_dir / "height_fingerprint.png"
    )
    make_candidate_table_image(
        pr.sdrl_result, pr.evidence_scores, output_dir / "verification_table.png"
    )
