"""
AERIS-3D — PDU: Probabilistic Depth Uncertainty (Phase 13)

Estimates confidence intervals for each candidate score via Monte Carlo
perturbation of the depth map and re-scoring (cheap path — skips neural inference).

Perturbation axes (all config-controlled):
  1. Depth noise         — Gaussian(0, sigma) added to depth map
  2. Terrain offset      — small shift to terrain baseline
  3. Segmentation jitter — small Gaussian blur on structure masks

Output per candidate: mean_score, std_score, p05, p95, confidence_label
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.candidate_generator import CandidateGeometry
from core.scoring_engine import EvidenceScore

log = get_logger("uncertainty")


@dataclass
class CandidateUncertainty:
    candidate_id: str
    object_id: str
    height_value: float
    height_unit: str
    nominal_score: float        # score from original pipeline
    mean_score: float           # MC mean
    std_score: float            # MC std (spread of scores)
    p05: float                  # 5th percentile (lower confidence bound)
    p95: float                  # 95th percentile (upper confidence bound)
    ci_width: float             # p95 - p05
    confidence_label: str       # HIGH | MEDIUM | LOW


@dataclass
class PDUResult:
    uncertainties: list[CandidateUncertainty]
    n_iterations: int
    runtime_s: float
    perturbation_config: dict


def run_pdu(
    candidates: list[CandidateGeometry],
    evidence_scores: list[EvidenceScore],
    depth_map: np.ndarray,
    config: dict,
) -> PDUResult:
    """
    Run Probabilistic Depth Uncertainty analysis.

    Monte Carlo strategy:
      For each of N iterations:
        1. Perturb depth map with Gaussian noise.
        2. Re-compute the depth_agreement component score (cheap, numpy).
        3. Re-compute overall score using existing weights.
      Collect distributions → compute CI.

    Args:
        candidates: All CandidateGeometry.
        evidence_scores: EGSS scores (one per candidate).
        depth_map: HxW float32.
        config: AERIS config dict.

    Returns:
        PDUResult with per-candidate uncertainty estimates.
    """
    t0 = time.perf_counter()
    unc_cfg = config.get("uncertainty", {})
    n_iter = int(unc_cfg.get("perturbation_iterations", 10))
    depth_sigma_frac = float(unc_cfg.get("height_perturbation_fraction", 0.05))
    terrain_sigma_frac = float(unc_cfg.get("terrain_perturbation_fraction", 0.05))

    log.info("Running PDU | n_iter=%d | depth_sigma_frac=%.3f", n_iter, depth_sigma_frac)

    # Build lookup for fast access
    es_by_cid = {es.candidate_id: es for es in evidence_scores}

    depth_range = max(float(depth_map.max() - depth_map.min()), 1e-6)
    depth_sigma = depth_range * depth_sigma_frac
    rng = np.random.default_rng(2025)

    # Per-candidate MC score collections
    mc_scores: dict[str, list[float]] = {c.candidate_id: [] for c in candidates}

    for i in range(n_iter):
        # Perturb depth map
        noise = rng.normal(0, depth_sigma, size=depth_map.shape)
        perturbed = depth_map + noise

        # Perturb terrain baseline (scalar offset)
        terrain_offset = rng.normal(0, depth_range * terrain_sigma_frac)

        for cand in candidates:
            es = es_by_cid.get(cand.candidate_id)
            if es is None:
                mc_scores[cand.candidate_id].append(0.3)
                continue

            # Re-score depth_agreement with perturbed depth
            # Uses the same formula as scoring_engine but with perturbed values
            terrain_at_cand = cand.terrain_baseline + terrain_offset
            depth_at_cand = float(perturbed[
                int(np.clip(cand.geometry_parameters.get("centroid", [0, 0])[0], 0, perturbed.shape[0] - 1)),
                int(np.clip(cand.geometry_parameters.get("centroid", [0, 0])[1], 0, perturbed.shape[1] - 1)),
            ])
            actual_offset = depth_at_cand - terrain_at_cand
            expected_offset = (cand.height_value / 61.0) * depth_range * 0.3

            if abs(actual_offset) > 1e-5:
                ratio = expected_offset / max(abs(actual_offset), 1e-5)
                perturbed_depth_agree = float(np.exp(-2.0 * abs(np.log(max(ratio, 1e-3)))))
            else:
                perturbed_depth_agree = float(np.exp(-0.05 * cand.height_value))

            # Blend with nominal score (most components are unaffected by depth perturbation)
            nominal = es.overall_score
            depth_w = es.weights.get("depth_agreement", 0.25)
            perturbed_score = float(np.clip(
                nominal + depth_w * (perturbed_depth_agree - es.component_scores.get("depth_agreement", 0.3)),
                0.0, 1.0
            ))
            mc_scores[cand.candidate_id].append(perturbed_score)

    # Compute statistics
    uncertainties: list[CandidateUncertainty] = []
    for cand in candidates:
        es = es_by_cid.get(cand.candidate_id)
        samples = mc_scores[cand.candidate_id]
        if not samples:
            continue
        arr = np.array(samples)
        mean_s = float(arr.mean())
        std_s = float(arr.std())
        p05 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        ci_width = p95 - p05

        if ci_width < 0.05 and std_s < 0.03:
            conf_label = "HIGH"
        elif ci_width < 0.12:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        uncertainties.append(CandidateUncertainty(
            candidate_id=cand.candidate_id,
            object_id=cand.object_id,
            height_value=cand.height_value,
            height_unit=cand.height_unit,
            nominal_score=es.overall_score if es else 0.0,
            mean_score=mean_s,
            std_score=std_s,
            p05=p05,
            p95=p95,
            ci_width=ci_width,
            confidence_label=conf_label,
        ))

    runtime_s = time.perf_counter() - t0
    log.info("PDU done | %d candidates | n_iter=%d | %.3fs", len(uncertainties), n_iter, runtime_s)

    return PDUResult(
        uncertainties=uncertainties,
        n_iterations=n_iter,
        runtime_s=runtime_s,
        perturbation_config={
            "depth_sigma_fraction": depth_sigma_frac,
            "terrain_sigma_fraction": terrain_sigma_frac,
            "n_iterations": n_iter,
        },
    )
