"""
AERIS-3D — Core Pipeline Orchestrator
Runs all phases in order, manages state, and returns a structured result.

Phases wired here:
  0  Config + Hardware
  1  Input Manager (called externally; InputData passed in)
  2  Depth Engine (called externally; DepthResult passed in)
  3  Structure Engine
  4  HCDC
  5  Candidate Generator
  6+7 CHF + ISCL
  8  Shadow Test
  9  Occlusion Test
  10 Terrain Solver
  11 EGSS
  12 SDRL

Phase 13+ (uncertainty, DSM, mesh, backend, frontend) in Prompt 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.candidate_generator import CandidateGeometry, generate_candidates
from core.counterfactual_height import CHFResult, run_chf_iscl
from core.depth_engine import DepthResult
from core.hcdc import HCDCResult, apply_hcdc
from core.input_manager import InputData
from core.logger import get_logger
from core.occlusion_test import OcclusionResult, run_occlusion_test
from core.scoring_engine import EvidenceScore, run_egss
from core.self_disproof import SDRLResult, run_sdrl
from core.shadow_test import ShadowTestResult, run_shadow_test
from core.structure_engine import StructureEngine, StructureResult
from core.terrain_solver import TerrainResult, solve_terrain

log = get_logger("pipeline")

# Stage labels for frontend progress updates
PIPELINE_STAGES = [
    "Loading",
    "Preprocessing",
    "Depth inference",
    "Structure analysis",
    "Candidate generation",
    "Counterfactual testing",
    "Verification",
    "Self-disproof",
    "Uncertainty",
    "DSM",
    "Mesh",
    "Complete",
]


@dataclass
class PipelineResult:
    """Complete result object from the AERIS-3D pipeline."""

    # Inputs
    input_data: InputData
    # Phase 2
    depth_result: DepthResult
    # Phase 3
    structure_result: StructureResult
    # Phase 4
    hcdc_result: HCDCResult
    # Phase 10 (run before candidates for terrain baseline)
    terrain_result: TerrainResult
    # Phase 5
    candidates: list[CandidateGeometry]
    # Phase 6+7
    chf_result: CHFResult
    # Phase 8
    shadow_result: ShadowTestResult
    # Phase 9
    occlusion_result: OcclusionResult
    # Phase 11
    evidence_scores: list[EvidenceScore]
    # Phase 12
    sdrl_result: SDRLResult

    total_runtime_s: float
    phase_runtimes: dict[str, float] = field(default_factory=dict)
    scale_mode: str = "RELATIVE"

    # Convenience accessors
    @property
    def best_per_object(self) -> dict:
        """Best-supported candidate per object: {object_id: EvidenceScore}"""
        return self.sdrl_result.survivors

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_rejected(self) -> int:
        return sum(len(v) for v in self.sdrl_result.rejected.values())

    @property
    def n_survived(self) -> int:
        return len(self.sdrl_result.survivors)


def run_pipeline(
    input_data: InputData,
    depth_result: DepthResult,
    config: dict,
) -> PipelineResult:
    """
    Run the full AERIS-3D analysis pipeline (Phases 3–12).

    Args:
        input_data: From input_manager.load_input()
        depth_result: From depth_engine.DepthEngine.estimate()
        config: Loaded AERIS config.

    Returns:
        PipelineResult with all intermediate and final results.
    """
    t_total = time.perf_counter()
    phase_times: dict[str, float] = {}
    image_rgb = input_data.image_rgb
    depth_map = depth_result.depth_map

    log.info("=== AERIS-3D Pipeline START ===")
    log.info(
        "Image: %dx%d | Depth: %dx%d | Scale: RELATIVE",
        image_rgb.shape[1],
        image_rgb.shape[0],
        depth_map.shape[1],
        depth_map.shape[0],
    )

    # ── Phase 3: Structure Analysis ───────────────────────────
    log.info("--- Phase 3: Structure Analysis ---")
    t3 = time.perf_counter()
    se = StructureEngine(config)
    structure_result = se.analyze(image_rgb, depth_map)
    phase_times["phase3_structure"] = time.perf_counter() - t3
    log.info(
        "Phase 3 done | %d buildings | quality=%.2f",
        len(structure_result.buildings),
        structure_result.segmentation_quality,
    )

    # ── Phase 4: HCDC ─────────────────────────────────────────
    log.info("--- Phase 4: HCDC ---")
    t4 = time.perf_counter()
    hcdc_result = apply_hcdc(depth_map, image_rgb, structure_result, config)
    corrected_depth = hcdc_result.corrected_depth
    phase_times["phase4_hcdc"] = time.perf_counter() - t4

    # ── Phase 10: Terrain Solver (before candidates) ───────────
    log.info("--- Phase 10: Terrain Solver ---")
    t10 = time.perf_counter()
    terrain_result = solve_terrain(
        corrected_depth, structure_result, input_data.georef, config
    )
    phase_times["phase10_terrain"] = time.perf_counter() - t10

    # ── Phase 5: Candidate Generation ─────────────────────────
    log.info("--- Phase 5: Candidate Generation ---")
    t5 = time.perf_counter()
    candidates = generate_candidates(
        structure_result, terrain_result, corrected_depth, config
    )
    phase_times["phase5_candidates"] = time.perf_counter() - t5
    log.info(
        "Phase 5 done | %d candidates across %d buildings",
        len(candidates),
        len(structure_result.buildings),
    )

    if not candidates:
        log.warning(
            "No candidates generated — pipeline cannot continue. Returning partial result."
        )
        # Return minimal result
        return _make_empty_result(
            input_data,
            depth_result,
            structure_result,
            hcdc_result,
            terrain_result,
            phase_times,
            t_total,
        )

    # ── Phase 6+7: CHF + ISCL ─────────────────────────────────
    log.info("--- Phase 6+7: CHF + ISCL ---")
    t6 = time.perf_counter()
    chf_result = run_chf_iscl(
        candidates, corrected_depth, structure_result, terrain_result, config
    )
    phase_times["phase6_7_chf_iscl"] = time.perf_counter() - t6

    # Build CHF score lookup for EGSS
    chf_score_lookup = {s.candidate_id: s for s in chf_result.candidate_scores}

    # ── Phase 8: Shadow Test ───────────────────────────────────
    log.info("--- Phase 8: Shadow Test ---")
    t8 = time.perf_counter()
    shadow_result = run_shadow_test(
        image_rgb,
        corrected_depth,
        structure_result,
        candidates,
        input_data.georef,
        config,
    )
    phase_times["phase8_shadow"] = time.perf_counter() - t8

    # ── Phase 9: Occlusion Test ────────────────────────────────
    log.info("--- Phase 9: Occlusion Test ---")
    t9 = time.perf_counter()
    occlusion_result = run_occlusion_test(
        candidates, corrected_depth, structure_result, config
    )
    phase_times["phase9_occlusion"] = time.perf_counter() - t9

    # ── Phase 11: EGSS ─────────────────────────────────────────
    log.info("--- Phase 11: EGSS ---")
    t11 = time.perf_counter()
    evidence_scores = run_egss(
        candidates,
        chf_score_lookup,
        shadow_result,
        occlusion_result,
        terrain_result,
        structure_result,
        config,
    )
    phase_times["phase11_egss"] = time.perf_counter() - t11

    # ── Phase 12: SDRL ─────────────────────────────────────────
    log.info("--- Phase 12: SDRL ---")
    t12 = time.perf_counter()
    sdrl_result = run_sdrl(candidates, evidence_scores, config)
    phase_times["phase12_sdrl"] = time.perf_counter() - t12

    total_runtime = time.perf_counter() - t_total
    log.info("=== AERIS-3D Pipeline COMPLETE | %.2fs ===", total_runtime)
    log.info(
        "  Buildings: %d | Candidates: %d | Survived: %d | Rejected: %d",
        len(structure_result.buildings),
        len(candidates),
        len(sdrl_result.survivors),
        sum(len(v) for v in sdrl_result.rejected.values()),
    )

    return PipelineResult(
        input_data=input_data,
        depth_result=depth_result,
        structure_result=structure_result,
        hcdc_result=hcdc_result,
        terrain_result=terrain_result,
        candidates=candidates,
        chf_result=chf_result,
        shadow_result=shadow_result,
        occlusion_result=occlusion_result,
        evidence_scores=evidence_scores,
        sdrl_result=sdrl_result,
        total_runtime_s=total_runtime,
        phase_runtimes=phase_times,
        scale_mode=terrain_result.scale_mode,
    )


def _make_empty_result(
    input_data,
    depth_result,
    structure_result,
    hcdc_result,
    terrain_result,
    phase_times,
    t_total,
) -> PipelineResult:
    """Return a partial result when no candidates are generated."""
    from core.candidate_generator import CandidateGeometry
    from core.counterfactual_height import CHFResult
    from core.occlusion_test import OcclusionResult
    from core.scoring_engine import EvidenceScore
    from core.self_disproof import SDRLResult
    from core.shadow_test import ShadowTestResult

    return PipelineResult(
        input_data=input_data,
        depth_result=depth_result,
        structure_result=structure_result,
        hcdc_result=hcdc_result,
        terrain_result=terrain_result,
        candidates=[],
        chf_result=CHFResult([], {}, {}, 0.0),
        shadow_result=ShadowTestResult(
            False, False, 0.0, None, None, None, 0.0, {}, 0.0, 0.0
        ),
        occlusion_result=OcclusionResult({}, {}, {}, 0.0),
        evidence_scores=[],
        sdrl_result=SDRLResult({}, {}, [], {}, 0.0),
        total_runtime_s=time.perf_counter() - t_total,
        phase_runtimes=phase_times,
        scale_mode="RELATIVE",
    )
