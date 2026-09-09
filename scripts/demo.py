#!/usr/bin/env python3
"""
AERIS-3D — Demo CLI
Usage: python scripts/demo.py --input <image_path> [--config <config.yaml>] [--output-dir <dir>]

Gates tested here:
  Gate 1 — image loads
  Gate 2 — depth map produced

Outputs (written to --output-dir, default: data/outputs/):
  input_preview.png    — resized RGB preview
  depth.png            — colorized depth (inferno colormap)
  initial_depth.png    — depth overlay on input image
  results.json         — structured pipeline metadata

If no --input is provided (or file not found), a synthetic test image is
auto-generated so the acceptance test can run on a clean install.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so 'core.*' imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.logger import get_logger
from core.config_loader import load_config, config_version_hash
from core.hardware import get_hardware
from core.input_manager import load_input
from core.depth_engine import DepthEngine


log = get_logger("demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AERIS-3D Demo — Single-image depth estimation and 3D reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input image (JPG/PNG/GeoTIFF). "
             "If omitted, a synthetic test image is auto-generated.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/outputs",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable depth caching (force re-inference).",
    )
    return parser.parse_args()


def generate_synthetic_test_image(output_path: Path) -> Path:
    """
    Create a small synthetic RGB image (aerial-like scene with blocks and gradients)
    for testing when no real input is available.
    """
    import numpy as np
    from PIL import Image as PILImage  # type: ignore

    log.warning("No input image provided — generating synthetic test image: %s", output_path)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky-like gradient background
    for y in range(h):
        val = int(30 + (y / h) * 80)
        img[y, :] = [val + 20, val + 30, val + 60]

    # Simulate terrain blocks (buildings/structures for structural analysis)
    rng = np.random.default_rng(42)
    for _ in range(12):
        bx = rng.integers(20, w - 80)
        by = rng.integers(20, h - 80)
        bw = rng.integers(30, 90)
        bh = rng.integers(30, 90)
        color = rng.integers(80, 200, size=3).tolist()
        img[by:by+bh, bx:bx+bw] = color

    # Road-like stripes
    img[h//2 - 10:h//2 + 10, :] = [60, 60, 60]
    img[:, w//2 - 8:w//2 + 8] = [60, 60, 60]

    # Vegetation patch (green tones)
    img[350:450, 50:180] = [40, 120, 50]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img).save(output_path)
    log.info("Synthetic test image written: %s (%dx%d)", output_path, w, h)
    return output_path


def colorize_depth(normalized_depth: "np.ndarray") -> "np.ndarray":
    """Apply inferno colormap to a [0,1] depth map → HxWx3 uint8."""
    import numpy as np
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    cmap = plt.get_cmap("inferno")
    rgba = cmap(normalized_depth)           # HxWx4 float
    rgb = (rgba[:, :, :3] * 255).astype("uint8")
    return rgb


def blend_overlay(image_rgb: "np.ndarray", overlay: "np.ndarray", alpha: float = 0.55) -> "np.ndarray":
    """Alpha-blend overlay (HxWx3 uint8) onto image_rgb."""
    import numpy as np
    base = image_rgb.astype(np.float32)
    over = overlay.astype(np.float32)
    blended = base * (1 - alpha) + over * alpha
    return blended.clip(0, 255).astype("uint8")


def save_image(arr: "np.ndarray", path: Path) -> None:
    from PIL import Image as PILImage  # type: ignore
    PILImage.fromarray(arr).save(path)
    log.info("Saved: %s", path)


def run_demo(args: argparse.Namespace) -> int:
    """Main demo routine. Returns 0 on success, 1 on failure."""
    pipeline_start = time.perf_counter()

    # ── Load config ──────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error("Failed to load config: %s", exc)
        return 1

    if args.no_cache:
        cfg.setdefault("depth", {})["cache_enabled"] = False
        log.info("Cache disabled by --no-cache flag.")

    cfg_hash = config_version_hash(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Hardware probe ───────────────────────────────────────
    hw = get_hardware()
    log.info("Runtime environment | device=%s (%s) | CPUs=%d",
             hw.device, hw.device_name, hw.cpu_count)

    # ── Resolve input ────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error("Input file not found: %s", input_path)
            return 1
    else:
        synthetic_path = Path("data/input/synthetic_test.jpg")
        input_path = generate_synthetic_test_image(synthetic_path)

    # ── Gate 1: Load input ───────────────────────────────────
    log.info("=== GATE 1: Loading input ===")
    try:
        input_data = load_input(input_path, cfg)
    except Exception as exc:
        log.error("Input loading failed: %s", exc, exc_info=True)
        return 1

    # Save input preview
    preview_path = output_dir / "input_preview.png"
    # Resize to max 1024 for preview
    from PIL import Image as PILImage  # type: ignore
    import numpy as np
    pil_preview = PILImage.fromarray(input_data.image_rgb)
    max_px = 1024
    if max(pil_preview.size) > max_px:
        pil_preview.thumbnail((max_px, max_px), PILImage.LANCZOS)
    pil_preview.save(preview_path)
    log.info("Gate 1 PASSED — input preview saved: %s", preview_path)

    # ── Gate 2: Depth estimation ─────────────────────────────
    log.info("=== GATE 2: Depth estimation ===")
    try:
        engine = DepthEngine(cfg)
        depth_result = engine.estimate(
            image_rgb=input_data.image_rgb,
            input_hash=input_data.input_hash,
            config_hash=cfg_hash,
        )
    except Exception as exc:
        log.error("Depth estimation failed: %s", exc, exc_info=True)
        return 1

    # Validate depth output
    dm = depth_result.depth_map
    nd = depth_result.normalized_depth
    if dm is None or dm.size == 0:
        log.error("Gate 2 FAILED — depth_map is empty.")
        return 1
    if dm.dtype != np.float32:
        log.error("Gate 2 FAILED — depth_map dtype is %s, expected float32.", dm.dtype)
        return 1
    if nd.min() < -0.01 or nd.max() > 1.01:
        log.warning("normalized_depth out of [0,1]: min=%.4f max=%.4f", nd.min(), nd.max())

    # Save depth.png
    depth_colorized = colorize_depth(nd)
    save_image(depth_colorized, output_dir / "depth.png")

    # Save initial_depth.png (overlay)
    # Resize image_rgb to match depth map if needed
    if input_data.image_rgb.shape[:2] != nd.shape:
        pil_src = PILImage.fromarray(input_data.image_rgb)
        pil_src = pil_src.resize((nd.shape[1], nd.shape[0]), PILImage.LANCZOS)
        rgb_for_overlay = np.array(pil_src)
    else:
        rgb_for_overlay = input_data.image_rgb

    overlay = blend_overlay(rgb_for_overlay, depth_colorized, alpha=0.5)
    save_image(overlay, output_dir / "initial_depth.png")

    log.info(
        "Gate 2 PASSED — depth_map shape=%s dtype=%s range=[%.4f,%.4f] model=%s device=%s cache=%s runtime=%.2fs",
        dm.shape, dm.dtype, float(dm.min()), float(dm.max()),
        depth_result.depth_metadata.model_name,
        depth_result.depth_metadata.device,
        depth_result.depth_metadata.cache_hit,
        depth_result.depth_metadata.runtime_s,
    )

    # ── Write results.json ───────────────────────────────────
    total_runtime = time.perf_counter() - pipeline_start
    meta = depth_result.depth_metadata
    results = {
        "aeris3d_version": "0.1.0",
        "pipeline_phase": "Phase0-2_Baseline",
        "gates_passed": ["Gate1_Input", "Gate2_Depth"],
        "input": {
            "file": str(input_path),
            "format": input_data.file_format.value,
            "original_size": {
                "height": input_data.preprocessing_meta.original_height,
                "width": input_data.preprocessing_meta.original_width,
            },
            "preprocessed_size": {
                "height": input_data.preprocessing_meta.preprocessed_height,
                "width": input_data.preprocessing_meta.preprocessed_width,
            },
            "georef_present": input_data.georef is not None,
            "input_hash": input_data.input_hash,
        },
        "depth": {
            "model": meta.model_name,
            "backend": meta.model_backend,
            "device": meta.device,
            "depth_map_shape": list(dm.shape),
            "depth_range": {
                "min": float(dm.min()),
                "max": float(dm.max()),
                "mean": float(dm.mean()),
            },
            "relative_only": meta.relative_only,
            "domain_warning": meta.domain_warning,
            "cache_hit": meta.cache_hit,
            "runtime_s": meta.runtime_s,
        },
        "hardware": {
            "device": hw.device,
            "device_name": hw.device_name,
            "cpu_count": hw.cpu_count,
            "torch_version": hw.torch_version,
            "platform": hw.platform,
        },
        "outputs": {
            "input_preview": str(output_dir / "input_preview.png"),
            "depth": str(output_dir / "depth.png"),
            "initial_depth": str(output_dir / "initial_depth.png"),
            "results_json": str(output_dir / "results.json"),
        },
        "total_runtime_s": total_runtime,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Results JSON written: %s", results_path)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AERIS-3D Demo — Phase 0-2 Complete")
    print("=" * 60)
    print(f"  Input:   {input_path}")
    print(f"  Format:  {input_data.file_format.value}")
    print(f"  Georef:  {'YES — CRS: ' + (input_data.georef.crs if input_data.georef else '') if input_data.georef else 'NO'}")
    print(f"  Model:   {meta.model_name} on {meta.device}")
    print(f"  Depth:   RELATIVE only (domain warning logged)")
    print(f"  Cache:   {'HIT' if meta.cache_hit else 'MISS'}")
    print(f"  Runtime: {meta.runtime_s:.2f}s depth | {total_runtime:.2f}s total")
    print(f"  Outputs: {output_dir}/")
    print("    - input_preview.png")
    print("    - depth.png")
    print("    - initial_depth.png")
    print("    - results.json")
    print("=" * 60)
    print("Gate 1 ✓  Gate 2 ✓")
    print()

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_demo(args))
