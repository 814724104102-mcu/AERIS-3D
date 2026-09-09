"""
AERIS-3D — OOF: Occlusion-Order Falsification (Phase 9)

Builds an approximate depth-layer visibility graph and checks whether
each candidate's geometry produces a plausible front/behind ordering.

Algorithm (image-space approximation):
1. Stratify the depth map into N layers (near → far).
2. For each pair of overlapping regions, determine their depth order.
3. For each candidate: does the candidate height imply a contradiction
   in the expected occlusion ordering?
4. Return occlusion_score per candidate and a machine-readable visibility graph.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.structure_engine import StructureResult
from core.candidate_generator import CandidateGeometry

log = get_logger("occlusion_test")


@dataclass
class OcclusionResult:
    candidate_occlusion_scores: dict[str, float]  # candidate_id → score 0-1
    visibility_graph: dict                         # simplified ordering graph
    contradictions: dict[str, list[str]]           # candidate_id → list of contradiction descriptions
    runtime_s: float


def run_occlusion_test(
    candidates: list[CandidateGeometry],
    depth_map: np.ndarray,
    structure_result: StructureResult,
    config: dict,
) -> OcclusionResult:
    """
    Run Occlusion-Order Falsification.

    Args:
        candidates: All candidates.
        depth_map: HxW float32 corrected depth.
        structure_result: StructureResult.
        config: AERIS config dict.

    Returns:
        OcclusionResult with per-candidate scores and contradiction lists.
    """
    t0 = time.perf_counter()
    oc_cfg = config.get("occlusion", {})
    n_layers = int(oc_cfg.get("depth_layer_count", 5))
    contr_thresh = float(oc_cfg.get("contradiction_threshold", 0.3))

    log.info("Running occlusion test | layers=%d | regions=%d",
             n_layers, len(structure_result.regions))

    regions = structure_result.regions
    if not regions:
        empty = {c.candidate_id: 0.5 for c in candidates}
        return OcclusionResult(
            candidate_occlusion_scores=empty,
            visibility_graph={},
            contradictions={c.candidate_id: [] for c in candidates},
            runtime_s=0.0,
        )

    # ── 1. Assign each region a depth layer ────────────────────
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_range = max(depth_max - depth_min, 1e-6)
    layer_size = depth_range / n_layers

    region_layers: dict[str, int] = {}
    for r in regions:
        # Closer to camera = higher depth value = lower layer index (front)
        # Depth layer: 0 = furthest (background), n_layers-1 = closest (foreground)
        norm_depth = (r.depth_mean - depth_min) / depth_range
        layer = int(np.clip(norm_depth * n_layers, 0, n_layers - 1))
        region_layers[r.object_id] = layer

    # ── 2. Build simple visibility graph (which regions are in front) ──
    visibility_graph = {}
    for r_a in regions:
        for r_b in regions:
            if r_a.object_id >= r_b.object_id:
                continue
            # Do their bounding boxes overlap?
            if _bboxes_overlap(r_a.bbox, r_b.bbox):
                layer_a = region_layers[r_a.object_id]
                layer_b = region_layers[r_b.object_id]
                if layer_a != layer_b:
                    front = r_a.object_id if layer_a > layer_b else r_b.object_id
                    back = r_b.object_id if layer_a > layer_b else r_a.object_id
                    key = f"{front}_in_front_of_{back}"
                    visibility_graph[key] = {
                        "front": front,
                        "back": back,
                        "layer_diff": abs(layer_a - layer_b),
                    }

    # ── 3. Score each candidate for occlusion plausibility ────
    candidate_scores: dict[str, float] = {}
    contradictions: dict[str, list[str]] = {}

    for cand in candidates:
        obj_id = cand.object_id
        h_val = cand.height_value
        contrs = []

        # Expected layer for this candidate: taller → should be in a higher (closer) layer
        region = next((r for r in regions if r.object_id == obj_id), None)
        if region is None:
            candidate_scores[cand.candidate_id] = 0.5
            contradictions[cand.candidate_id] = []
            continue

        expected_layer = region_layers.get(obj_id, 0)

        # Check: if candidate height is very large, it should be among the closest objects
        # Contradiction: candidate height >> but region is in a far layer
        max_layer = max(region_layers.values()) if region_layers else 0
        normalized_layer = expected_layer / max(max_layer, 1)  # 0=far, 1=near

        # Higher candidate height → expect higher layer (closer to camera)
        height_fraction = np.clip(h_val / 60.0, 0, 1)  # normalize to [0,1]
        layer_height_consistency = 1.0 - abs(normalized_layer - height_fraction) * 0.7

        # Check pair-wise contradictions
        for vis_key, vis_data in visibility_graph.items():
            if obj_id not in (vis_data["front"], vis_data["back"]):
                continue
            is_front = vis_data["front"] == obj_id
            other_id = vis_data["back"] if is_front else vis_data["front"]
            other_region = next((r for r in regions if r.object_id == other_id), None)
            if other_region is None:
                continue

            # If we're supposedly in front (closer), our candidate height should
            # be consistent with having more depth than the object behind us
            if is_front and h_val < 3.0:
                contrs.append(
                    f"Candidate {cand.candidate_id} height={h_val:.1f} is very low "
                    f"but {obj_id} appears closer than {other_id} in depth layers."
                )

        score = float(np.clip(layer_height_consistency, 0.2, 1.0))
        if contrs:
            score *= max(0.5, 1.0 - 0.2 * len(contrs))

        candidate_scores[cand.candidate_id] = score
        contradictions[cand.candidate_id] = contrs

    runtime_s = time.perf_counter() - t0
    log.info("Occlusion test done | graph_edges=%d | %.3fs",
             len(visibility_graph), runtime_s)

    return OcclusionResult(
        candidate_occlusion_scores=candidate_scores,
        visibility_graph=visibility_graph,
        contradictions=contradictions,
        runtime_s=runtime_s,
    )


def _bboxes_overlap(bbox_a: tuple, bbox_b: tuple) -> bool:
    """Check if two (y1, x1, y2, x2) bounding boxes overlap."""
    ay1, ax1, ay2, ax2 = bbox_a
    by1, bx1, by2, bx2 = bbox_b
    return not (ay2 < by1 or by2 < ay1 or ax2 < bx1 or bx2 < ax1)
