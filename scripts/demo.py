#!/usr/bin/env python3
"""
AERIS-3D — Demo CLI (Phases 0–2)
Usage: python scripts/demo.py --input <image_path> [--config <config.yaml>] [--output-dir <dir>]

Gates tested:
  Gate 1 — image loads
  Gate 2 — depth map produced

Outputs:
  input_preview.png        — resized RGB preview
  depth.png                — colorized depth (inferno)
  initial_depth.png        — depth overlay on input
  results.json             — structured pipeline metadata
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config_loader import config_version_hash, load_config
from core.depth_engine import DepthEngine
from core.hardware import get_hardware
from core.input_manager import load_input
from core.logger import get_logger

log = get_logger("demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AERIS-3D Demo — Single-image 3D reconstruction (Phases 0-2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input image (JPG/PNG/GeoTIFF). Auto-generates if omitted.",
    )
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--output-dir", "-o", type=str, default="data/outputs")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def generate_synthetic_test_image(output_path: Path) -> Path:
    from PIL import Image as PILImage

    log.warning("No input image — generating synthetic test image: %s", output_path)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(30 + (y / h) * 80)
        img[y, :] = [val + 20, val + 30, val + 60]
    rng = np.random.default_rng(42)
    for _ in range(12):
        bx, by = rng.integers(20, w - 80), rng.integers(20, h - 80)
        bw, bh = rng.integers(30, 90), rng.integers(30, 90)
        img[by : by + bh, bx : bx + bw] = rng.integers(80, 200, size=3).tolist()
    img[h // 2 - 10 : h // 2 + 10, :] = [60, 60, 60]
    img[:, w // 2 - 8 : w // 2 + 8] = [60, 60, 60]
    img[350:450, 50:180] = [40, 120, 50]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img).save(output_path)
    return output_path


def colorize_depth(nd: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("inferno")
    rgba = cmap(nd)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    from PIL import Image as PILImage

    PILImage.fromarray(arr).save(path)
    log.info("Saved: %s", path)


def run_demo(args: argparse.Namespace) -> int:
    pipeline_start = time.perf_counter()

    # ── Load config ───────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error("Config load failed: %s", exc)
        return 1

    if args.no_cache:
        cfg.setdefault("depth", {})["cache_enabled"] = False

    cfg_hash = config_version_hash(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hw = get_hardware()
    log.info(
        "Hardware: device=%s (%s) CPUs=%d", hw.device, hw.device_name, hw.cpu_count
    )

    # ── Resolve input ─────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error("Input not found: %s", input_path)
            return 1
    else:
        input_path = generate_synthetic_test_image(
            Path("data/input/synthetic_test.jpg")
        )

    # ── Gate 1: Load input ────────────────────────────────────
    log.info("=== GATE 1: Loading input ===")
    try:
        input_data = load_input(input_path, cfg)
    except Exception as exc:
        log.error("Input loading failed: %s", exc, exc_info=True)
        return 1

    from PIL import Image as PILImage

    pil_prev = PILImage.fromarray(input_data.image_rgb)
    if max(pil_prev.size) > 1024:
        pil_prev.thumbnail((1024, 1024), PILImage.LANCZOS)
    pil_prev.save(output_dir / "input_preview.png")
    log.info("Gate 1 PASSED")

    # ── Gate 2: Depth estimation ──────────────────────────────
    log.info("=== GATE 2: Depth estimation ===")
    try:
        engine = DepthEngine(cfg)
        depth_result = engine.estimate(
            input_data.image_rgb, input_data.input_hash, cfg_hash
        )
    except Exception as exc:
        log.error("Depth failed: %s", exc, exc_info=True)
        return 1

    nd = depth_result.normalized_depth
    save_image(colorize_depth(nd), output_dir / "depth.png")
    
    # Overlay
    pil_rgb = PILImage.fromarray(input_data.image_rgb)
    pil_dep = PILImage.fromarray(colorize_depth(nd))
    pil_dep = pil_dep.resize(pil_rgb.size, PILImage.BILINEAR)
    blended = PILImage.blend(pil_rgb, pil_dep, 0.5)
    blended.save(output_dir / "initial_depth.png")
    log.info(
        "Gate 2 PASSED | model=%s device=%s cache=%s runtime=%.2fs",
        depth_result.depth_metadata.model_name,
        depth_result.depth_metadata.device,
        depth_result.depth_metadata.cache_hit,
        depth_result.depth_metadata.runtime_s,
    )

    # ── Write results.json ────────────────────────────────────
    total_runtime = time.perf_counter() - pipeline_start
    results = {
        "aeris3d_version": "0.2.0",
        "pipeline_phase": "Phase0-2_AERIS_Core",
        "gates_passed": [
            "Gate1_Input",
            "Gate2_Depth",
        ],
        "input": {
            "file": str(input_path),
            "format": input_data.file_format.value,
            "original_size": {
                "height": input_data.preprocessing_meta.original_height,
                "width": input_data.preprocessing_meta.original_width,
            },
            "georef_present": input_data.georef is not None,
        },
        "depth": {
            "model": depth_result.depth_metadata.model_name,
            "device": depth_result.depth_metadata.device,
            "relative_only": depth_result.depth_metadata.relative_only,
            "runtime_s": depth_result.depth_metadata.runtime_s,
        },
        "total_runtime_s": round(total_runtime, 3),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved: %s", output_dir / "results.json")

    print("\n=================================================================")
    print("AERIS-3D Demo — Phase 0-2 Complete")
    print("=================================================================")
    print(f"  Input:    {input_path}")
    print(f"  Georef:   {'YES' if input_data.georef else 'NO'}")
    print(f"  Model:    {depth_result.depth_metadata.model_name} on {depth_result.depth_metadata.device}")
    print(f"  Runtime:  {total_runtime:.2f}s total")
    print("\n  Outputs:")
    print("    ✓ input_preview.png")
    print("    ✓ depth.png")
    print("    ✓ initial_depth.png")
    print("    ✓ results.json")
    print("=================================================================")
    print("  Gate1 ✓  Gate2 ✓\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_demo(parse_args()))
