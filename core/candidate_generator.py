"""
AERIS-3D — Candidate Generator (Phase 5)
For each detected building structure, generates multiple candidate heights
using the configured range, step size, and adaptive narrowing.

Output: List[CandidateGeometry] — each carries candidate_id, object_id,
        height_m (relative or metric), terrain_baseline, geometry parameters.

Heights are labeled RELATIVE unless a metric anchor is available from TerrainResult.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.structure_engine import StructureResult, StructureRegion
from core.terrain_solver import TerrainResult

log = get_logger("candidate_generator")


@dataclass
class CandidateGeometry:
    candidate_id: str
    object_id: str

    # Height in depth-units (RELATIVE) or metres (ANCHORED_METRIC)
    height_value: float
    height_unit: str                    # "RELATIVE_DEPTH_UNITS" | "METRES_ESTIMATED"

    terrain_baseline: float             # terrain depth at object location
    geometry_parameters: dict = field(default_factory=dict)

    # Set by scoring engine later
    consistency_score: float = 0.0
    component_scores: dict = field(default_factory=dict)
    status: str = "TESTING"             # TESTING | SURVIVED | REJECTED
    rejection_reason: Optional[str] = None
    iteration_eliminated: Optional[int] = None


def generate_candidates(
    structure_result: StructureResult,
    terrain_result: TerrainResult,
    depth_map: np.ndarray,
    config: dict,
) -> list[CandidateGeometry]:
    """
    Generate candidate heights for every detected building structure.

    For each building region:
      1. Estimate depth range attributable to height (depth_offset range)
      2. Generate N candidate heights spanning [min_height, max_height]
      3. Optionally narrow the range using adaptive depth-cue (if depth signal is strong)

    Returns flat list of all candidates across all objects.
    """
    t0 = time.perf_counter()
    cg_cfg = config.get("candidate_generator", {})

    # Config parameters
    min_h = float(cg_cfg.get("min_height_m", 3.0))
    max_h = float(cg_cfg.get("max_height_m", 60.0))
    step_h = float(cg_cfg.get("step_m", 3.0))
    top_k = int(cg_cfg.get("top_k", 5))
    adaptive = bool(cg_cfg.get("adaptive_range", True))
    scale_mode = terrain_result.scale_mode

    buildings = structure_result.buildings
    log.info("Generating candidates | buildings=%d | range=[%.1f, %.1f] step=%.1f | scale=%s",
             len(buildings), min_h, max_h, step_h, scale_mode)

    if not buildings:
        log.warning("No building regions detected — no candidates generated.")
        return []

    all_candidates: list[CandidateGeometry] = []
    cand_counter = 0

    # Depth range of the full scene — used to scale relative height estimates
    depth_scene_range = float(depth_map.max() - depth_map.min())
    if depth_scene_range < 1e-6:
        depth_scene_range = 1.0

    for region in buildings:
        obj_id = region.object_id
        terrain_at_object = float(terrain_result.terrain_surface[
            int(region.centroid[0]), int(region.centroid[1])
        ])
        depth_offset = region.depth_relative_to_terrain  # positive = above terrain

        # ── Adaptive range narrowing ────────────────────────────
        if adaptive and abs(depth_offset) > 0.005:
            # The depth offset gives us a hint about the actual height range.
            # Map depth offset → expected height range.
            # Without metric scale: depth_offset is in raw depth units.
            # Scale: 1 depth unit ≈ scale_factor × physical height.
            # We don't know scale_factor, so we generate a range ±50% around
            # the depth-suggested centroid to bracket the true height.
            depth_suggested_fraction = abs(depth_offset) / depth_scene_range
            # Heuristic: map [0,1] depth fraction to [min_h, max_h] linearly
            depth_suggested_h = min_h + depth_suggested_fraction * (max_h - min_h)
            adaptive_min = max(min_h, depth_suggested_h * 0.4)
            adaptive_max = min(max_h, depth_suggested_h * 2.0)
            adaptive_step = max(step_h * 0.5, (adaptive_max - adaptive_min) / 8)
        else:
            adaptive_min = min_h
            adaptive_max = max_h
            adaptive_step = step_h

        candidate_heights = np.arange(adaptive_min, adaptive_max + adaptive_step * 0.1, adaptive_step)

        # Ensure at least 4 candidates for a meaningful rejection demonstration
        if len(candidate_heights) < 4:
            candidate_heights = np.linspace(min_h, max_h, max(5, top_k + 2))

        log.debug("Object %s | depth_offset=%.4f | adaptive=[%.1f, %.1f] step=%.1f | %d candidates",
                  obj_id, depth_offset, adaptive_min, adaptive_max, adaptive_step,
                  len(candidate_heights))

        for h_val in candidate_heights:
            cand_id = f"H{cand_counter:04d}"
            bbox = region.bbox  # (y1, x1, y2, x2)

            # Geometry parameters: simplified extruded box
            geom = {
                "footprint_bbox": list(bbox),
                "centroid": list(region.centroid),
                "area_px": region.area_px,
                "expected_depth_at_top": float(terrain_at_object + h_val / (max_h + 1) * depth_scene_range),
                "aspect_ratio": _bbox_aspect_ratio(bbox),
            }

            unit = (
                "METRES_ESTIMATED" if scale_mode == "ANCHORED_METRIC"
                else "RELATIVE_DEPTH_UNITS"
            )

            cand = CandidateGeometry(
                candidate_id=cand_id,
                object_id=obj_id,
                height_value=float(h_val),
                height_unit=unit,
                terrain_baseline=terrain_at_object,
                geometry_parameters=geom,
            )
            all_candidates.append(cand)
            cand_counter += 1

    runtime_s = time.perf_counter() - t0
    log.info("Candidates generated | total=%d across %d buildings | %.3fs",
             len(all_candidates), len(buildings), runtime_s)
    return all_candidates


def _bbox_aspect_ratio(bbox: tuple) -> float:
    """Height/Width ratio of a bounding box (y1, x1, y2, x2)."""
    y1, x1, y2, x2 = bbox
    bh = max(y2 - y1, 1)
    bw = max(x2 - x1, 1)
    return float(bh / bw)
