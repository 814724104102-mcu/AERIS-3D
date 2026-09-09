"""
AERIS-3D — Terrain Solver: TBCS (Phase 10)
Separates local terrain elevation from object-height-above-terrain.

SCIENTIFIC HONESTY:
  Without an external elevation anchor (GeoTIFF pixel size, GSD,
  or a reference DEM), all output is RELATIVE — never metric.
  If an anchor is available, outputs are labeled ANCHORED_METRIC.
  The two are never blurred together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.structure_engine import StructureResult

log = get_logger("terrain_solver")

RELATIVE_LABEL = "RELATIVE"
ANCHORED_LABEL = "ANCHORED_METRIC"


@dataclass
class TerrainResult:
    terrain_surface: np.ndarray  # HxW float32 — estimated local ground elevation
    object_height_map: np.ndarray  # HxW float32 — height above terrain per pixel
    scale_mode: str  # RELATIVE | ANCHORED_METRIC
    pixels_per_meter: Optional[float]  # None if RELATIVE
    terrain_depth_level: float  # depth value reference for terrain
    terrain_roughness: float  # std of terrain surface within ground pixels
    runtime_s: float


def solve_terrain(
    corrected_depth: np.ndarray,
    structure_result: StructureResult,
    georef,  # GeoRef or None
    config: dict,
) -> TerrainResult:
    """
    Separate terrain baseline from object heights above it.

    Uses the lowest-depth (furthest from camera / ground-level) pixels
    in structurally flat regions as the terrain reference.

    Args:
        corrected_depth: HxW float32 from HCDC.
        structure_result: StructureResult from StructureEngine.
        georef: GeoRef if available (provides pixel size for metric scale).
        config: AERIS config dict.

    Returns:
        TerrainResult with terrain surface and object height map.
    """
    t0 = time.perf_counter()
    terrain_cfg = config.get("terrain", {})
    log.info("Solving terrain baseline | georef=%s", "YES" if georef else "NO")

    from scipy.ndimage import gaussian_filter, percentile_filter

    h, w = corrected_depth.shape

    # ── 1. Build ground mask ────────────────────────────────────
    from core.structure_engine import CLASSES

    label_map = structure_result.label_map
    road_idx = CLASSES.index("road")
    terrain_idx = CLASSES.index("terrain")
    ground_mask = (label_map == road_idx) | (label_map == terrain_idx)

    if ground_mask.sum() < 50:
        # Fallback: use lowest 20th percentile of depth as terrain
        threshold = float(np.percentile(corrected_depth, 20))
        ground_mask = corrected_depth <= threshold
        log.warning(
            "Insufficient ground pixels — using depth percentile as terrain proxy."
        )

    # ── 2. Terrain surface estimation ──────────────────────────
    # Fill ground depths, interpolate over building/other regions
    terrain_surface = corrected_depth.copy()

    # Set non-ground pixels to NaN, then inpaint
    terrain_only = np.full((h, w), np.nan, dtype=np.float32)
    terrain_only[ground_mask] = corrected_depth[ground_mask]

    # Simple inpainting: fill NaN with a spatially smoothed terrain estimate
    terrain_surface = _inpaint_nan(terrain_only, sigma=max(h, w) / 8.0)

    terrain_depth_level = float(np.nanmedian(terrain_only[ground_mask]))
    terrain_roughness = float(np.nanstd(terrain_only[ground_mask]))

    # ── 3. Object height map ────────────────────────────────────
    # Height above terrain = corrected_depth - terrain_surface
    # Positive values = above terrain (closer to camera)
    object_height_map = (corrected_depth - terrain_surface).astype(np.float32)
    object_height_map = np.clip(object_height_map, 0, None)  # no negative heights

    # ── 4. Determine scale mode ─────────────────────────────────
    pixels_per_meter: Optional[float] = None
    scale_mode = RELATIVE_LABEL

    if georef is not None:
        px_size = min(abs(georef.pixel_size_x), abs(georef.pixel_size_y))
        if px_size > 0:
            # px_size is in degrees or metres depending on CRS
            # For metric CRS: pixels_per_meter = 1.0 / px_size
            # We report it but don't claim metric height without full calibration
            pixels_per_meter = 1.0 / px_size if px_size < 1.0 else None
            if pixels_per_meter is not None:
                scale_mode = ANCHORED_LABEL
                log.info(
                    "Terrain scale: %.2f px/m from GeoTIFF CRS pixel size",
                    pixels_per_meter,
                )
            else:
                log.info(
                    "GeoTIFF pixel size %.6f degrees — metric scale requires proj. Not anchoring.",
                    px_size,
                )

    if scale_mode == RELATIVE_LABEL:
        log.info("Terrain mode: RELATIVE — no metric anchor available.")

    runtime_s = time.perf_counter() - t0
    log.info(
        "Terrain solved | mode=%s | terrain_level=%.4f | roughness=%.4f | %.3fs",
        scale_mode,
        terrain_depth_level,
        terrain_roughness,
        runtime_s,
    )

    return TerrainResult(
        terrain_surface=terrain_surface,
        object_height_map=object_height_map,
        scale_mode=scale_mode,
        pixels_per_meter=pixels_per_meter,
        terrain_depth_level=terrain_depth_level,
        terrain_roughness=terrain_roughness,
        runtime_s=runtime_s,
    )


def _inpaint_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    """
    Simple NaN inpainting via weighted Gaussian smoothing.
    Missing pixels are filled using a distance-weighted average of known values.
    """
    from scipy.ndimage import gaussian_filter

    mask_valid = ~np.isnan(arr)
    filled = arr.copy()
    filled[~mask_valid] = 0.0

    weights = mask_valid.astype(np.float32)
    smooth_num = gaussian_filter(filled, sigma=sigma)
    smooth_den = gaussian_filter(weights, sigma=sigma)

    result = np.where(smooth_den > 1e-6, smooth_num / smooth_den, 0.0)
    # Preserve known values exactly
    result[mask_valid] = arr[mask_valid]
    return result.astype(np.float32)
