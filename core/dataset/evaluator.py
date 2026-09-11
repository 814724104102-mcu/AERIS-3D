"""
Baseline Evaluator for AERIS-3D.
Computes zero-shot metrics (MAE, RMSE) of the pretrained backbone against the dataset.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from core.depth_engine import DepthEngine
from core.config_loader import load_config
from core.logger import get_logger

log = get_logger(__name__)

def evaluate_baseline(ds, test_indices: list, output_dir: Path):
    """
    Runs the pretrained depth engine on test samples and computes baseline metrics.
    """
    log.info(f"Starting baseline evaluation on {len(test_indices)} test samples...")
    config = load_config()
    engine = DepthEngine(config)
    
    # We only care about relative error since scale calibration happens later in AERIS
    maes = []
    rmses = []
    
    # Track metrics by land cover class if GAMUS provides it, but here we just do global
    for idx in test_indices:
        sample = ds[idx]
        
        # Load image and target
        if "image" not in sample or ("dsm" not in sample and "height" not in sample):
            continue
            
        img_pil = sample["image"]
        # GAMUS provides normalized DSM in "dsm" or "height"
        target_key = "dsm" if "dsm" in sample else "height"
        target_img = sample[target_key]
        
        # Convert to numpy arrays
        # The dataset might return PIL images or numpy arrays directly
        if hasattr(img_pil, "convert"):
            img_arr = np.array(img_pil.convert("RGB"))
        else:
            img_arr = np.array(img_pil)
            
        if hasattr(target_img, "convert"):
            # L mode or F mode
            target_arr = np.array(target_img).astype(np.float32)
        else:
            target_arr = np.array(target_img).astype(np.float32)
            
            # Run inference
        try:
            depth_result = engine.estimate(img_arr, f"gamus_test_{idx}", "eval")
            norm_depth = depth_result.depth_normalized
            model_name = depth_result.depth_metadata.model_name
        except Exception as e:
            log.warning(f"Depth inference failed on sample {idx}: {e}")
            continue
        
        # Compute metrics (Relative scale: we normalize both to 0-1 for a fair baseline comparison)
        # Note: If target is all zeros, skip to avoid division by zero
        target_min, target_max = target_arr.min(), target_arr.max()
        if target_max > target_min:
            norm_target = (target_arr - target_min) / (target_max - target_min)
            
            # Ensure sizes match
            if norm_depth.shape != norm_target.shape:
                import cv2
                norm_depth = cv2.resize(norm_depth, (norm_target.shape[1], norm_target.shape[0]))
                
            error = norm_depth - norm_target
            maes.append(np.mean(np.abs(error)))
            rmses.append(np.sqrt(np.mean(error**2)))
            
    if not maes:
        log.warning("No valid reference depth/DSM targets found in the dataset subset. Quantitative evaluation skipped per rules.")
        metrics = {
            "MAE_normalized": "SKIPPED_NO_GROUND_TRUTH",
            "RMSE_normalized": "SKIPPED_NO_GROUND_TRUTH",
            "evaluated_samples": 0,
            "model": engine.depth_cfg.get("model", "depth_anything_v2_small")
        }
    else:
        metrics = {
            "MAE_normalized": float(np.mean(maes)),
            "RMSE_normalized": float(np.mean(rmses)),
            "evaluated_samples": len(maes),
            "model": model_name
        }
        
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    if isinstance(metrics['MAE_normalized'], float):
        log.info(f"Baseline metrics computed: MAE={metrics['MAE_normalized']:.4f}, RMSE={metrics['RMSE_normalized']:.4f}")
    else:
        log.info(f"Baseline metrics computed: MAE={metrics['MAE_normalized']}, RMSE={metrics['RMSE_normalized']}")
        
    return metrics_path
