"""
AERIS-3D — Mesh Builder (Phase 15)

Converts the DSM (relative surface model) into a 3D triangle mesh and exports
as GLB (GL Binary) for Three.js rendering.

Algorithm:
  1. Downsample DSM to mesh_resolution × mesh_resolution grid.
  2. Apply vertical_exaggeration from config (default 1.5×) for visual clarity.
  3. Build triangle mesh: 2 triangles per grid cell (regular grid).
  4. Assign vertex colors from the RGB image (mapped to DSM grid).
  5. Export with trimesh → GLB.

The GLB is served by the FastAPI backend to the Three.js frontend.

Scale is RELATIVE — vertex Z coordinates are in depth units, not metres.
The export metadata carries scale_mode warning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.logger import get_logger

log = get_logger("mesh_builder")


@dataclass
class MeshResult:
    glb_path: Path
    vertex_count: int
    face_count: int
    scale_mode: str
    vertical_exaggeration: float
    runtime_s: float
    warning: Optional[str] = None


def build_mesh(
    dsm: np.ndarray,
    image_rgb: np.ndarray,
    scale_mode: str,
    output_dir: Path,
    config: dict,
) -> MeshResult:
    """
    Build a triangle mesh from the DSM and export as GLB.

    Args:
        dsm: HxW float32 surface model (RELATIVE depth).
        image_rgb: HxW3 uint8 RGB image for vertex colors.
        scale_mode: "RELATIVE" | "ANCHORED_METRIC"
        output_dir: Directory to write mesh.glb.
        config: AERIS config dict.

    Returns:
        MeshResult with GLB path, vertex/face counts, and metadata.
    """
    t0 = time.perf_counter()
    mesh_cfg = config.get("mesh", {})
    resolution = int(mesh_cfg.get("resolution", 256))
    v_exag = float(mesh_cfg.get("vertical_exaggeration", 1.5))
    export_obj = mesh_cfg.get("export_obj", False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building mesh | resolution=%d | v_exag=%.1f", resolution, v_exag)

    try:
        import trimesh
    except ImportError:
        log.error("trimesh not installed — mesh export skipped.")
        return MeshResult(
            glb_path=output_dir / "mesh.glb",
            vertex_count=0,
            face_count=0,
            scale_mode=scale_mode,
            vertical_exaggeration=v_exag,
            runtime_s=0.0,
            warning="trimesh not installed",
        )

    h_dsm, w_dsm = dsm.shape

    # ── 1. Resize DSM and image to mesh resolution ─────────────
    from PIL import Image as PILImage

    dsm_img = PILImage.fromarray(dsm)
    dsm_img = dsm_img.resize((resolution, resolution), PILImage.BILINEAR)
    dsm_small = np.array(dsm_img, dtype=np.float32)

    rgb_img = PILImage.fromarray(image_rgb)
    rgb_img = rgb_img.resize((resolution, resolution), PILImage.LANCZOS)
    rgb_small = np.array(rgb_img, dtype=np.uint8)

    # ── 2. Normalize DSM to [0, 1] range ──────────────────────
    d_min, d_max = dsm_small.min(), dsm_small.max()
    d_range = max(d_max - d_min, 1e-6)
    dsm_norm = (dsm_small - d_min) / d_range  # 0 = far, 1 = near/high

    # ── 3. Build vertex grid ───────────────────────────────────
    # X, Y in [-1, 1] (image plane)
    # Z = dsm_norm × vertical_exaggeration
    cols = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    rows = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    X, Y = np.meshgrid(cols, rows)  # both (resolution, resolution)
    Z = dsm_norm * v_exag

    # Flatten to vertex array
    vertices = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)

    # Vertex colors from RGB image (normalized to [0, 1])
    vertex_colors = rgb_small.reshape(-1, 3).astype(np.float32) / 255.0

    # ── 4. Build face indices (two triangles per grid cell) ────
    r = resolution
    idx = np.arange(r * r, dtype=np.int32).reshape(r, r)
    # Top-left, top-right, bottom-left, bottom-right of each cell
    tl = idx[:-1, :-1].ravel()
    tr = idx[:-1, 1:].ravel()
    bl = idx[1:, :-1].ravel()
    br = idx[1:, 1:].ravel()

    faces = np.concatenate(
        [
            np.stack([tl, tr, bl], axis=1),  # upper triangle
            np.stack([tr, br, bl], axis=1),  # lower triangle
        ],
        axis=0,
    )  # (2*(r-1)^2, 3)

    # ── 5. Build trimesh and export ────────────────────────────
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,  # skip auto-processing (we've built it correctly)
    )

    # Optional decimation for large meshes
    max_faces = int(mesh_cfg.get("decimation_face_count", 50000))
    if len(mesh.faces) > max_faces:
        try:
            # trimesh 5.x + fast_simplification: use face_count keyword
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
            log.info("Mesh decimated to %d faces", len(mesh.faces))
        except Exception as exc:
            log.warning("Mesh decimation failed (%s) — using full mesh.", exc)



    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    # Export GLB
    glb_path = output_dir / "mesh.glb"
    glb_bytes = mesh.export(file_type="glb")
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)
    log.info(
        "Saved mesh.glb: %d vertices, %d faces, %.1f KB",
        n_verts,
        n_faces,
        len(glb_bytes) / 1024,
    )

    # Optional OBJ export
    if export_obj:
        obj_path = output_dir / "mesh.obj"
        mesh.export(str(obj_path))
        log.info("Saved mesh.obj: %s", obj_path)

    runtime_s = time.perf_counter() - t0
    warning = (
        "Mesh Z-axis is in RELATIVE depth units — not metric elevation."
        if scale_mode == "RELATIVE"
        else None
    )
    log.info("Mesh built | %d verts | %d faces | %.3fs", n_verts, n_faces, runtime_s)

    return MeshResult(
        glb_path=glb_path,
        vertex_count=n_verts,
        face_count=n_faces,
        scale_mode=scale_mode,
        vertical_exaggeration=v_exag,
        runtime_s=runtime_s,
        warning=warning,
    )
