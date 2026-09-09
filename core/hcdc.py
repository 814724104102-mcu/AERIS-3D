"""
AERIS-3D — HCDC: Hypothesis-Conditioned Depth Correction (Phase 4)
Refines raw monocular depth using structural information.

The depth engine produces an INITIAL hypothesis. HCDC identifies regions where
the initial depth is inconsistent with structural evidence and applies a
lightweight correction. It does NOT claim to produce metric depth.

Key operations:
- Flag pixels where depth gradient >> image gradient (depth discontinuity without
  a visible edge → likely model error in that region)
- Edge-aware smoothing within homogeneous regions
- Region-aware correction: snap region interior to a consistent depth level
- Output: corrected_depth, correction_map, structural_boundary_map
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.logger import get_logger
from core.structure_engine import StructureResult

log = get_logger("hcdc")


@dataclass
class HCDCResult:
    corrected_depth: np.ndarray  # HxW float32 (corrected relative depth)
    correction_map: np.ndarray  # HxW float32  magnitude of correction applied
    structural_boundary_map: (
        np.ndarray
    )  # HxW float32  combined image+depth boundary strength
    flagged_fraction: float  # fraction of pixels flagged as inconsistent
    runtime_s: float


def apply_hcdc(
    depth_map: np.ndarray,
    image_rgb: np.ndarray,
    structure_result: StructureResult,
    config: dict,
) -> HCDCResult:
    """
    Apply Hypothesis-Conditioned Depth Correction.

    Args:
        depth_map: HxW float32 raw depth from depth engine.
        image_rgb: HxW3 uint8 original image.
        structure_result: Output of StructureEngine.
        config: AERIS config dict.

    Returns:
        HCDCResult with corrected depth and diagnostic maps.
    """
    t0 = time.perf_counter()
    cfg = config.get("hcdc", {})
    sigma = cfg.get("edge_aware_sigma", 2.0)
    grad_thresh = cfg.get("gradient_threshold", 0.15)
    consistency_radius = int(cfg.get("local_consistency_radius", 5))
    correction_strength = float(cfg.get("correction_strength", 0.4))

    log.info(
        "Applying HCDC | sigma=%.1f | grad_thresh=%.2f | strength=%.2f",
        sigma,
        grad_thresh,
        correction_strength,
    )

    import cv2
    from scipy.ndimage import gaussian_filter, uniform_filter

    h, w = depth_map.shape

    # ── 1. Normalize depth to [0, 1] for gradient comparison ──
    d_min, d_max = depth_map.min(), depth_map.max()
    d_range = max(d_max - d_min, 1e-6)
    d_norm = (depth_map - d_min) / d_range  # [0,1]

    # ── 2. Compute image gradient magnitude ────────────────────
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx_img = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_img = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    img_grad = np.sqrt(gx_img**2 + gy_img**2)
    img_grad_norm = img_grad / max(img_grad.max(), 1e-6)

    # ── 3. Compute depth gradient magnitude ────────────────────
    gx_dep = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy_dep = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
    dep_grad = np.sqrt(gx_dep**2 + gy_dep**2)
    dep_grad_norm = dep_grad / max(dep_grad.max(), 1e-6)

    # ── 4. Structural boundary map (union of both gradients) ───
    structural_boundary_map = np.maximum(img_grad_norm, dep_grad_norm)

    # ── 5. Flag inconsistent pixels ────────────────────────────
    # Inconsistency: high depth gradient where image gradient is LOW
    # (depth changes abruptly, but the image shows no edge there)
    inconsistent = (dep_grad_norm > grad_thresh) & (img_grad_norm < grad_thresh * 0.5)
    flagged_fraction = float(inconsistent.mean())

    # ── 6. Edge-aware smoothing ────────────────────────────────
    # Smooth depth within structurally consistent regions
    # Use the structural boundary as an edge-stopping weight
    edge_weight = np.exp(-structural_boundary_map * 5.0)
    smoothed = gaussian_filter(depth_map * edge_weight, sigma=sigma)
    weight_sum = gaussian_filter(edge_weight, sigma=sigma)
    smoothed = smoothed / np.maximum(weight_sum, 1e-6)

    # ── 7. Region-interior depth leveling ──────────────────────
    # For each detected building region, replace interior depth with
    # a smoothed region-mean (buildings should have consistent roof depth)
    corrected = depth_map.copy()
    for region in structure_result.buildings:
        if region.area_px < 200:
            continue
        interior_mask = region.mask.copy()
        # Erode mask slightly to avoid edge contamination
        interior_vals = depth_map[interior_mask]
        if interior_vals.size == 0:
            continue
        region_target = float(np.median(interior_vals))
        region_current = depth_map[interior_mask]
        # Blend toward the region median
        corrected[interior_mask] = (
            region_current * (1 - correction_strength)
            + region_target * correction_strength
        )

    # ── 8. Blend corrected with smoothed for flagged pixels ────
    correction_map = np.abs(corrected - depth_map)
    corrected[inconsistent] = (
        corrected[inconsistent] * (1 - correction_strength * 0.5)
        + smoothed[inconsistent] * correction_strength * 0.5
    )

    final_correction_map = np.abs(corrected - depth_map).astype(np.float32)
    runtime_s = time.perf_counter() - t0

    log.info(
        "HCDC done | flagged=%.1f%% | max_correction=%.4f | %.3fs",
        flagged_fraction * 100,
        float(final_correction_map.max()),
        runtime_s,
    )

    return HCDCResult(
        corrected_depth=corrected.astype(np.float32),
        correction_map=final_correction_map,
        structural_boundary_map=structural_boundary_map.astype(np.float32),
        flagged_fraction=flagged_fraction,
        runtime_s=runtime_s,
    )
