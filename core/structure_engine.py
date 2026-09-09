"""
AERIS-3D — Structure Engine (Phase 3)
Extracts building/road/vegetation/water/terrain/unknown regions from an image.

Backend: "classical" (Canny + SLIC + heuristic classifier)
Degrades gracefully when segmentation quality is poor — never raises on bad input.

Classes: building, road, vegetation, water, terrain, unknown
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.logger import get_logger

log = get_logger("structure_engine")

CLASSES = ["building", "road", "vegetation", "water", "terrain", "unknown"]


@dataclass
class StructureRegion:
    object_id: str
    label: str  # one of CLASSES
    mask: np.ndarray  # HxW bool (original image coords)
    bbox: tuple  # (y1, x1, y2, x2) in original image coords
    area_px: int
    centroid: tuple  # (cy, cx)
    depth_mean: float
    depth_std: float
    depth_relative_to_terrain: float  # positive = above terrain (closer to camera)
    color_mean: np.ndarray  # [R, G, B] float
    confidence: float  # classifier confidence 0-1


@dataclass
class StructureResult:
    regions: list[StructureRegion]
    edge_map: np.ndarray  # HxW uint8 Canny edges
    boundary_map: np.ndarray  # HxW float32  depth boundary strength
    label_map: np.ndarray  # HxW int8  per-pixel class index (-1=unclassified)
    terrain_depth_level: float  # median depth of terrain/road pixels (reference level)
    segmentation_quality: float  # 0-1 estimated quality
    backend_used: str
    runtime_s: float
    num_regions: int
    warning: Optional[str] = None

    @property
    def buildings(self) -> list[StructureRegion]:
        return [r for r in self.regions if r.label == "building"]

    @property
    def roads(self) -> list[StructureRegion]:
        return [r for r in self.regions if r.label == "road"]


class StructureEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.struct_cfg = config.get("structure", {})

    def analyze(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
    ) -> StructureResult:
        """
        Run structural analysis on an RGB image + depth map.

        Returns:
            StructureResult with detected regions, edge map, and label map.
        """
        t0 = time.perf_counter()
        backend = self.struct_cfg.get("segmentation_backend", "classical")
        log.info(
            "Running structural analysis | backend=%s | image=%dx%d",
            backend,
            image_rgb.shape[1],
            image_rgb.shape[0],
        )

        try:
            result = self._run_classical(image_rgb, depth_map)
        except Exception as exc:
            log.error(
                "Structure analysis failed: %s — returning empty result.",
                exc,
                exc_info=True,
            )
            h, w = image_rgb.shape[:2]
            result = StructureResult(
                regions=[],
                edge_map=np.zeros((h, w), dtype=np.uint8),
                boundary_map=np.zeros((h, w), dtype=np.float32),
                label_map=np.full((h, w), -1, dtype=np.int8),
                terrain_depth_level=float(np.median(depth_map)),
                segmentation_quality=0.0,
                backend_used="failed",
                runtime_s=time.perf_counter() - t0,
                num_regions=0,
                warning=f"Structural analysis failed: {exc}",
            )

        log.info(
            "Structure analysis done | backend=%s | regions=%d | buildings=%d | quality=%.2f | %.2fs",
            result.backend_used,
            result.num_regions,
            len(result.buildings),
            result.segmentation_quality,
            result.runtime_s,
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Classical backend: Canny + SLIC + heuristic classifier
    # ─────────────────────────────────────────────────────────

    def _run_classical(
        self, image_rgb: np.ndarray, depth_map: np.ndarray
    ) -> StructureResult:
        import cv2
        from scipy.ndimage import gaussian_filter
        from skimage.measure import label as cc_label
        from skimage.measure import regionprops
        from skimage.segmentation import slic

        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]
        cfg = self.struct_cfg

        # ── Edge detection ────────────────────────────────────
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lo = cfg.get("canny_low_threshold", 50)
        hi = cfg.get("canny_high_threshold", 150)
        edges = cv2.Canny(gray, lo, hi)

        # ── Depth boundary map ────────────────────────────────
        d_norm = (depth_map - depth_map.min()) / max(
            depth_map.max() - depth_map.min(), 1e-6
        )
        gx = cv2.Sobel(d_norm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d_norm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        boundary_map = np.sqrt(gx**2 + gy**2).astype(np.float32)

        # ── SLIC superpixels ──────────────────────────────────
        n_seg = cfg.get("slic_n_segments", 200)
        compactness = cfg.get("slic_compactness", 10.0)
        try:
            segments = slic(
                image_rgb,
                n_segments=n_seg,
                compactness=compactness,
                sigma=1.0,
                start_label=0,
                channel_axis=-1,
            )
        except Exception as exc:
            log.warning("SLIC failed (%s) — using grid segmentation fallback.", exc)
            segments = self._grid_segments(h, w, n_seg)

        # ── Heuristic classification per superpixel ────────────
        num_superpixels = segments.max() + 1
        sp_labels = np.full(num_superpixels, -1, dtype=np.int8)
        sp_confidence = np.zeros(num_superpixels, dtype=np.float32)

        for sp_id in range(num_superpixels):
            mask = segments == sp_id
            if mask.sum() < cfg.get("min_region_area_px", 100):
                continue
            color = image_rgb[mask].mean(axis=0)  # [R, G, B]
            depth_vals = depth_map[mask]
            lbl, conf = _classify_superpixel(color, depth_vals)
            sp_labels[sp_id] = CLASSES.index(lbl)
            sp_confidence[sp_id] = conf

        # ── Build per-pixel label map ─────────────────────────
        label_map = np.full((h, w), -1, dtype=np.int8)
        for sp_id in range(num_superpixels):
            if sp_labels[sp_id] >= 0:
                label_map[segments == sp_id] = sp_labels[sp_id]

        # ── Terrain depth level (reference) ────────────────────
        terrain_idx = CLASSES.index("terrain")
        road_idx = CLASSES.index("road")
        ground_mask = (label_map == terrain_idx) | (label_map == road_idx)
        if ground_mask.sum() > 100:
            terrain_depth_level = float(np.median(depth_map[ground_mask]))
        else:
            # Fallback: use 10th percentile of depth as terrain estimate
            terrain_depth_level = float(np.percentile(depth_map, 10))

        # ── Merge superpixels into connected building regions ──
        building_idx = CLASSES.index("building")
        building_mask = label_map == building_idx
        cc, n_cc = _connected_components(building_mask)

        min_area = cfg.get("min_region_area_px", 100)
        regions: list[StructureRegion] = []
        obj_counter = 0

        for cc_id in range(1, n_cc + 1):
            obj_mask = cc == cc_id
            area = int(obj_mask.sum())
            if area < min_area:
                continue

            ys, xs = np.where(obj_mask)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            cy = float(ys.mean())
            cx = float(xs.mean())

            d_vals = depth_map[obj_mask]
            d_mean = float(d_vals.mean())
            d_std = float(d_vals.std())
            d_rel = d_mean - terrain_depth_level

            color_mean = image_rgb[obj_mask].mean(axis=0).astype(np.float32)

            region = StructureRegion(
                object_id=f"obj_{obj_counter:03d}",
                label="building",
                mask=obj_mask,
                bbox=(y1, x1, y2, x2),
                area_px=area,
                centroid=(cy, cx),
                depth_mean=d_mean,
                depth_std=d_std,
                depth_relative_to_terrain=d_rel,
                color_mean=color_mean,
                confidence=0.7,
            )
            regions.append(region)
            obj_counter += 1

        # Also add non-building regions as informational regions
        for lbl_idx, lbl_name in enumerate(CLASSES):
            if lbl_name in {"building", "unknown"}:
                continue
            lbl_mask = label_map == lbl_idx
            if lbl_mask.sum() < min_area * 2:
                continue
            cc2, n_cc2 = _connected_components(lbl_mask)
            for cc_id in range(1, min(n_cc2 + 1, 6)):  # max 5 per class
                obj_mask = cc2 == cc_id
                if obj_mask.sum() < min_area * 2:
                    continue
                ys, xs = np.where(obj_mask)
                y1, y2 = int(ys.min()), int(ys.max())
                x1, x2 = int(xs.min()), int(xs.max())
                d_vals = depth_map[obj_mask]
                region = StructureRegion(
                    object_id=f"obj_{obj_counter:03d}",
                    label=lbl_name,
                    mask=obj_mask,
                    bbox=(y1, x1, y2, x2),
                    area_px=int(obj_mask.sum()),
                    centroid=(float(ys.mean()), float(xs.mean())),
                    depth_mean=float(d_vals.mean()),
                    depth_std=float(d_vals.std()),
                    depth_relative_to_terrain=float(d_vals.mean())
                    - terrain_depth_level,
                    color_mean=image_rgb[obj_mask].mean(axis=0).astype(np.float32),
                    confidence=0.6,
                )
                regions.append(region)
                obj_counter += 1

        # ── Segmentation quality estimate ─────────────────────
        classified_px = (label_map >= 0).sum()
        quality = float(classified_px / (h * w))

        runtime_s = time.perf_counter() - t0
        return StructureResult(
            regions=regions,
            edge_map=edges,
            boundary_map=boundary_map,
            label_map=label_map,
            terrain_depth_level=terrain_depth_level,
            segmentation_quality=quality,
            backend_used="classical_canny_slic",
            runtime_s=runtime_s,
            num_regions=len(regions),
        )

    @staticmethod
    def _grid_segments(h: int, w: int, n: int) -> np.ndarray:
        """Fallback: uniform grid segmentation."""
        side = int(np.sqrt(n))
        rows = np.linspace(0, h, side + 1, dtype=int)
        cols = np.linspace(0, w, side + 1, dtype=int)
        out = np.zeros((h, w), dtype=np.int32)
        idx = 0
        for i in range(side):
            for j in range(side):
                out[rows[i] : rows[i + 1], cols[j] : cols[j + 1]] = idx
                idx += 1
        return out


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────


def _classify_superpixel(
    color: np.ndarray,  # [R, G, B] float
    depth_vals: np.ndarray,
) -> tuple[str, float]:
    """
    Heuristic multi-class classification of a superpixel.
    Uses color statistics and depth variation.
    Returns (class_label, confidence).
    """
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    brightness = (r + g + b) / 3.0
    d_std = float(depth_vals.std())
    d_mean = float(depth_vals.mean())

    # NDVI-style: vegetation has high green relative to red and blue
    green_excess = g - max(r, b)
    # Water: dark, blue-dominant
    blue_excess = b - max(r, g)
    # Road: dark grey (low brightness, low saturation)
    saturation = float(np.std([r, g, b]))
    # Building: moderate brightness, non-green, varied depth

    if green_excess > 20 and g > 60:
        return "vegetation", min(0.9, 0.6 + green_excess / 100)
    if blue_excess > 15 and brightness < 120:
        return "water", min(0.9, 0.6 + blue_excess / 80)
    if brightness < 80 and saturation < 20:
        return "road", 0.75
    if brightness < 100 and saturation < 30:
        return "terrain", 0.65
    if d_std > 0.05 and brightness > 80:
        return "building", 0.70
    if brightness > 150:
        return "building", 0.60
    return "terrain", 0.50


def _connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components. Returns (label_array, n_components)."""
    try:
        import cv2

        mask_u8 = binary_mask.astype(np.uint8)
        n, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        return labels, n - 1  # subtract background
    except Exception:
        from scipy.ndimage import label as sp_label

        labeled, n = sp_label(binary_mask)
        return labeled, n
