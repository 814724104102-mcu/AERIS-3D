"""
AERIS-3D — Demo CLI (Phases 0–12)
Usage: python scripts/demo.py --input <image_path> [--config <config.yaml>] [--output-dir <dir>]
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
from core.pipeline import run_pipeline

log = get_logger("demo")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AERIS-3D Demo — Single-image 3D reconstruction (Phases 0-12)",
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

def save_plot(pipeline_result, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # We create a visualization showing height vs consistency graph
    chf_result = pipeline_result.chf_result
    sdrl_result = pipeline_result.sdrl_result
    
    for obj_id, best_score in sdrl_result.survivors.items():
        if obj_id not in chf_result.height_fingerprint:
            continue
        fp = chf_result.height_fingerprint[obj_id]
        heights = fp["heights"]
        scores = fp["scores"]
        unit = fp.get("height_unit", "units")
        
        plt.figure(figsize=(10, 6))
        plt.plot(heights, scores, marker='o', linestyle='-', color='blue', label='CHF Score')
        
        # Mark SURVIVED and REJECTED
        survivor_h = best_score.height_value
        plt.axvline(survivor_h, color='green', linestyle='--', label=f'Best Supported ({survivor_h:.1f})')
        
        rejected_cands = sdrl_result.rejected.get(obj_id, [])
        rejected_heights = [cand.height_value for cand in rejected_cands]
        rejected_scores = [cand.overall_score for cand in rejected_cands] # Wait, these are EGSS scores, let's plot those
        
        # Plot EGSS scores too
        # Find all EGSS scores for this object
        egss_scores = [score for score in pipeline_result.evidence_scores if score.object_id == obj_id]
        egss_h = [s.height_value for s in egss_scores]
        egss_s = [s.overall_score for s in egss_scores]
        
        # Sort for plotting
        sorted_egss = sorted(zip(egss_h, egss_s))
        plt.plot([x[0] for x in sorted_egss], [x[1] for x in sorted_egss], marker='s', linestyle='-', color='purple', label='EGSS Score')
        
        # Plot rejected
        if rejected_heights:
            plt.scatter(rejected_heights, rejected_scores, color='red', marker='x', s=100, label='Rejected', zorder=5)
            
        plt.scatter([survivor_h], [best_score.overall_score], color='green', marker='*', s=200, label='Survived', zorder=5)

        plt.title(f'Object {obj_id}: Height vs Consistency')
        plt.xlabel(f'Height ({unit})')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"height_consistency_{obj_id}.png")
        plt.close()

def run_demo(args: argparse.Namespace) -> int:
    pipeline_start = time.perf_counter()

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

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = generate_synthetic_test_image(Path("data/input/synthetic_test.jpg"))

    input_data = load_input(input_path, cfg)
    engine = DepthEngine(cfg)
    depth_result = engine.estimate(input_data.image_rgb, input_data.input_hash, cfg_hash)

    # RUN FULL PIPELINE
    pipeline_result = run_pipeline(input_data, depth_result, cfg)
    
    # Visualizations
    save_plot(pipeline_result, output_dir)
    
    # Save results.json
    results = {
        "pipeline_phase": "Phase0-12",
        "candidates_evaluated": pipeline_result.n_candidates,
        "n_survived": pipeline_result.n_survived,
        "n_rejected": pipeline_result.n_rejected,
        "phase_runtimes": pipeline_result.phase_runtimes,
        "total_runtime_s": pipeline_result.total_runtime_s,
        "objects": {}
    }
    
    for obj_id, survivor in pipeline_result.sdrl_result.survivors.items():
        rejected = pipeline_result.sdrl_result.rejected.get(obj_id, [])
        results["objects"][obj_id] = {
            "best_height": survivor.height_value,
            "best_score": survivor.overall_score,
            "status": "SURVIVED",
            "component_scores": survivor.component_scores,
            "rejected_candidates": [
                {
                    "candidate_id": r.candidate_id,
                    "height": r.height_value,
                    "score": r.overall_score,
                    "status": "REJECTED"
                } for r in rejected
            ]
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved: %s", output_dir / "results.json")

    return 0

if __name__ == "__main__":
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    exit_code = run_demo(parse_args())
    pr.disable()
    pr.dump_stats("pipeline.prof")
    print("Profiling saved to pipeline.prof")
    import pstats
    p = pstats.Stats('pipeline.prof')
    p.sort_stats('cumulative').print_stats(20)
    sys.exit(exit_code)
