"""
AERIS-3D — DSM/rDSM Exporter (Phase 14)

Exports the Digital Surface Model (DSM) and Relative DSM (rDSM).

DSM  = corrected depth (proxy for surface elevation — RELATIVE unless anchored)
rDSM = object height above terrain (from terrain_solver)

Export formats:
  - PNG  — colorized elevation map (terrain colormap)
  - NPZ  — raw numpy arrays (dsm, rdsm, metadata)
  - JSON — metadata + stats
  - GeoTIFF — only if CRS is available in georef; else skipped with warning

SCIENTIFIC HONESTY:
  All exports carry scale_mode metadata.
  GeoTIFF is skipped (not fabricated) when no CRS is available.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.terrain_solver import TerrainResult

log = get_logger("dsm_exporter")


@dataclass
class DSMExportResult:
    dsm: np.ndarray             # HxW float32 surface model
    rdsm: np.ndarray            # HxW float32 relative (height above terrain)
    scale_mode: str             # RELATIVE | ANCHORED_METRIC
    exported_files: list[str]
    stats: dict
    runtime_s: float


def export_dsm(
    corrected_depth: np.ndarray,
    terrain_result: TerrainResult,
    georef,                     # GeoRef or None
    output_dir: Path,
    config: dict,
) -> DSMExportResult:
    """
    Export DSM and rDSM to configured formats.

    Args:
        corrected_depth: HxW float32 from HCDC.
        terrain_result: TerrainResult with terrain_surface and object_height_map.
        georef: GeoRef or None.
        output_dir: Directory to write files.
        config: AERIS config dict.

    Returns:
        DSMExportResult with arrays and list of exported files.
    """
    t0 = time.perf_counter()
    dsm_cfg = config.get("dsm", {})
    export_png = dsm_cfg.get("export_png", True)
    export_npz = dsm_cfg.get("export_npz", True)
    export_geotiff = dsm_cfg.get("export_geotiff", True)
    export_json = dsm_cfg.get("export_json_metadata", True)
    colormap = dsm_cfg.get("colormap", "terrain")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dsm = corrected_depth.astype(np.float32)
    rdsm = terrain_result.object_height_map.astype(np.float32)
    scale_mode = terrain_result.scale_mode
    exported: list[str] = []

    log.info("Exporting DSM/rDSM | scale=%s | png=%s npz=%s geotiff=%s",
             scale_mode, export_png, export_npz, export_geotiff)

    # ── Stats ──────────────────────────────────────────────────
    dsm_stats = {
        "min": float(dsm.min()),
        "max": float(dsm.max()),
        "mean": float(dsm.mean()),
        "std": float(dsm.std()),
    }
    rdsm_stats = {
        "min": float(rdsm.min()),
        "max": float(rdsm.max()),
        "mean": float(rdsm.mean()),
        "std": float(rdsm.std()),
        "nonzero_fraction": float((rdsm > 0.01).mean()),
    }

    # ── PNG export ─────────────────────────────────────────────
    if export_png:
        _save_colorized(dsm, output_dir / "dsm.png", colormap, "DSM")
        _save_colorized(rdsm, output_dir / "rdsm.png", "hot", "rDSM")
        exported += ["dsm.png", "rdsm.png"]

    # ── NPZ export ─────────────────────────────────────────────
    if export_npz:
        npz_path = output_dir / "dsm.npz"
        np.savez_compressed(
            npz_path,
            dsm=dsm,
            rdsm=rdsm,
            terrain_surface=terrain_result.terrain_surface,
        )
        log.info("Saved: %s", npz_path)
        exported.append("dsm.npz")

    # ── JSON metadata ──────────────────────────────────────────
    if export_json:
        meta = {
            "scale_mode": scale_mode,
            "pixels_per_meter": terrain_result.pixels_per_meter,
            "terrain_depth_level": terrain_result.terrain_depth_level,
            "terrain_roughness": terrain_result.terrain_roughness,
            "dsm_stats": dsm_stats,
            "rdsm_stats": rdsm_stats,
            "shape": list(dsm.shape),
            "warning": (
                "All elevation values are RELATIVE — no metric anchor. "
                "Do not use as absolute elevations."
                if scale_mode == "RELATIVE" else None
            ),
        }
        meta_path = output_dir / "dsm_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Saved: %s", meta_path)
        exported.append("dsm_metadata.json")

    # ── GeoTIFF export ─────────────────────────────────────────
    if export_geotiff:
        if georef is None:
            log.warning("GeoTIFF export skipped — no CRS/georef available. "
                        "Fabricating coordinates is not allowed.")
        else:
            try:
                _save_geotiff(dsm, rdsm, georef, output_dir)
                exported += ["dsm.tif", "rdsm.tif"]
            except Exception as exc:
                log.error("GeoTIFF export failed: %s", exc)

    runtime_s = time.perf_counter() - t0
    log.info("DSM export done | files=%s | %.3fs", exported, runtime_s)

    return DSMExportResult(
        dsm=dsm,
        rdsm=rdsm,
        scale_mode=scale_mode,
        exported_files=exported,
        stats={"dsm": dsm_stats, "rdsm": rdsm_stats},
        runtime_s=runtime_s,
    )


def _save_colorized(arr: np.ndarray, path: Path, colormap: str, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr)

    cmap = plt.get_cmap(colormap)
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    PILImage.fromarray(rgb).save(path)
    log.info("Saved %s: %s (range=[%.4f, %.4f])", label, path, arr_min, arr_max)


def _save_geotiff(dsm: np.ndarray, rdsm: np.ndarray, georef, output_dir: Path) -> None:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    transform = from_bounds(
        georef.bounds[0], georef.bounds[1],
        georef.bounds[2], georef.bounds[3],
        dsm.shape[1], dsm.shape[0],
    )
    crs = CRS.from_string(georef.crs)

    for arr, fname, desc in [(dsm, "dsm.tif", "DSM"), (rdsm, "rdsm.tif", "rDSM")]:
        out_path = output_dir / fname
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(arr, 1)
            dst.update_tags(
                scale_mode="ANCHORED_METRIC",
                aeris3d_version="0.2.0",
            )
        log.info("Saved GeoTIFF %s: %s", desc, out_path)
