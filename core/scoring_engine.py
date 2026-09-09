"""
AERIS-3D — EGSS: Evidence-Weighted Geometric Survival Score (Phase 11)

Combines all evidence sources into a single interpretable per-candidate score
with EVIDENCE-ADAPTIVE weights:
  - Shadow unreliable → shadow_weight → 0
  - Segmentation weak → boundary_weight reduced
  - Elevation anchor present → terrain_weight increased

Weights live in configs/default.yaml. This scorer is INTERPRETABLE:
it returns component_scores, active_evidence, and weights so the UI
can explain every number.

NEVER a black box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.candidate_generator import CandidateGeometry
from core.counterfactual_height import CandidateScore as CHFScore
from core.logger import get_logger
from core.occlusion_test import OcclusionResult
from core.shadow_test import ShadowTestResult
from core.structure_engine import StructureResult
from core.terrain_solver import TerrainResult

log = get_logger("scoring_engine")


@dataclass
class EvidenceScore:
    candidate_id: str
    object_id: str
    height_value: float
    height_unit: str
    overall_score: float
    component_scores: dict[str, float]
    active_evidence: list[str]
    weights: dict[str, float]
    rejection_threshold: float
    passes_threshold: bool


def run_egss(
    candidates: list[CandidateGeometry],
    chf_scores: dict[str, CHFScore],  # candidate_id → CHFScore
    shadow_result: ShadowTestResult,
    occlusion_result: OcclusionResult,
    terrain_result: TerrainResult,
    structure_result: StructureResult,
    config: dict,
) -> list[EvidenceScore]:
    """
    Compute EGSS (Evidence-Weighted Geometric Survival Score) for all candidates.

    Args:
        candidates: All CandidateGeometry.
        chf_scores: Mapping candidate_id → CHFScore (from CHF module).
        shadow_result: ShadowTestResult.
        occlusion_result: OcclusionResult.
        terrain_result: TerrainResult.
        structure_result: StructureResult.
        config: AERIS config dict.

    Returns:
        List of EvidenceScore (one per candidate), ordered by overall_score desc.
    """
    t0 = time.perf_counter()
    sc_cfg = config.get("scoring", {})
    base_weights = sc_cfg.get("weights", {})
    rejection_threshold = float(sc_cfg.get("rejection_threshold", 0.35))

    log.info(
        "Running EGSS | candidates=%d | rejection_threshold=%.2f",
        len(candidates),
        rejection_threshold,
    )

    # ── Evidence-adaptive weight computation ────────────────────
    # Start from config base weights, then adjust based on evidence quality

    w_depth = float(base_weights.get("depth_agreement", 0.25))
    w_edge = float(base_weights.get("edge_overlap", 0.25))
    w_boundary = float(base_weights.get("segmentation_boundary", 0.15))
    w_shadow = float(base_weights.get("shadow", 0.10))
    w_occlusion = float(base_weights.get("occlusion", 0.10))
    w_terrain = float(base_weights.get("terrain", 0.10))
    w_struct = float(base_weights.get("structural_plausibility", 0.05))

    # Adapt shadow weight
    if not shadow_result.shadow_reliable:
        w_shadow_eff = 0.0
        log.info("EGSS: shadow weight set to 0 (unreliable)")
    else:
        w_shadow_eff = w_shadow * shadow_result.shadow_reliability

    # Adapt boundary weight based on segmentation quality
    seg_quality = structure_result.segmentation_quality
    if seg_quality < 0.3:
        w_boundary_eff = w_boundary * 0.3
        log.info("EGSS: boundary weight reduced (seg_quality=%.2f)", seg_quality)
    else:
        w_boundary_eff = w_boundary * seg_quality

    # Adapt terrain weight
    if terrain_result.scale_mode == "ANCHORED_METRIC":
        w_terrain_eff = w_terrain * 1.5
        log.info("EGSS: terrain weight boosted (ANCHORED_METRIC)")
    else:
        w_terrain_eff = w_terrain * 0.5

    # Redistribute weight from zeroed sources to CHF components
    total_w = (
        w_depth
        + w_edge
        + w_boundary_eff
        + w_shadow_eff
        + w_occlusion
        + w_terrain_eff
        + w_struct
    )
    if total_w < 1e-6:
        total_w = 1.0

    active_evidence = []
    if w_depth > 0:
        active_evidence.append("depth_agreement")
    if w_edge > 0:
        active_evidence.append("edge_overlap")
    if w_boundary_eff > 0:
        active_evidence.append("segmentation_boundary")
    if w_shadow_eff > 0:
        active_evidence.append("shadow")
    if w_occlusion > 0:
        active_evidence.append("occlusion")
    if w_terrain_eff > 0:
        active_evidence.append("terrain")
    if w_struct > 0:
        active_evidence.append("structural_plausibility")

    final_weights = {
        "depth_agreement": w_depth / total_w,
        "edge_overlap": w_edge / total_w,
        "segmentation_boundary": w_boundary_eff / total_w,
        "shadow": w_shadow_eff / total_w,
        "occlusion": w_occlusion / total_w,
        "terrain": w_terrain_eff / total_w,
        "structural_plausibility": w_struct / total_w,
    }

    log.info(
        "EGSS effective weights: %s", {k: f"{v:.3f}" for k, v in final_weights.items()}
    )

    results: list[EvidenceScore] = []

    for cand in candidates:
        cid = cand.candidate_id
        chf = chf_scores.get(cid)
        shadow_score = shadow_result.candidate_shadow_scores.get(cid, 0.5)
        occlusion_score = occlusion_result.candidate_occlusion_scores.get(cid, 0.5)

        # Structural plausibility (Gaussian around typical building heights)
        struct_plaus = float(np.exp(-0.002 * (cand.height_value - 15.0) ** 2))

        # Terrain consistency: does the candidate height agree with terrain height map?
        from core.structure_engine import CLASSES

        region = next(
            (r for r in structure_result.regions if r.object_id == cand.object_id), None
        )
        if region is not None:
            cy, cx = int(region.centroid[0]), int(region.centroid[1])
            cy = np.clip(cy, 0, terrain_result.object_height_map.shape[0] - 1)
            cx = np.clip(cx, 0, terrain_result.object_height_map.shape[1] - 1)
            terrain_height_px = float(terrain_result.object_height_map[cy, cx])
            depth_range = max(
                terrain_result.object_height_map.max()
                - terrain_result.object_height_map.min(),
                1e-6,
            )
            # Map depth height to a height-unit-comparable value
            depth_predicted_h = terrain_height_px / depth_range * 60.0
            terrain_score = float(
                np.exp(-0.01 * (cand.height_value - depth_predicted_h) ** 2)
            )
        else:
            terrain_score = 0.5

        # CHF component scores
        if chf is not None:
            depth_score = chf.depth_agreement
            edge_score = chf.edge_overlap
            boundary_score = chf.boundary_match
        else:
            depth_score = 0.3
            edge_score = 0.3
            boundary_score = 0.3

        component_scores = {
            "depth_agreement": depth_score,
            "edge_overlap": edge_score,
            "segmentation_boundary": boundary_score,
            "shadow": shadow_score,
            "occlusion": occlusion_score,
            "terrain": terrain_score,
            "structural_plausibility": struct_plaus,
        }

        overall = sum(component_scores[k] * final_weights[k] for k in final_weights)
        overall = float(np.clip(overall, 0.0, 1.0))

        results.append(
            EvidenceScore(
                candidate_id=cid,
                object_id=cand.object_id,
                height_value=cand.height_value,
                height_unit=cand.height_unit,
                overall_score=overall,
                component_scores=component_scores,
                active_evidence=active_evidence,
                weights=final_weights,
                rejection_threshold=rejection_threshold,
                passes_threshold=overall >= rejection_threshold,
            )
        )

    results.sort(key=lambda x: x.overall_score, reverse=True)
    n_pass = sum(1 for r in results if r.passes_threshold)
    runtime_s = time.perf_counter() - t0

    log.info(
        "EGSS done | %d/%d candidates pass threshold=%.2f | %.3fs",
        n_pass,
        len(results),
        rejection_threshold,
        runtime_s,
    )
    return results
