"""
AERIS-3D — Depth Engine (Phase 2)
Monocular depth estimation. Treats output as RELATIVE depth — NOT metric.

Backbone priority (configurable via configs/default.yaml → depth.model):
  1. Depth Anything V2 Small/Base/Large  (via HuggingFace Transformers)
  2. MiDaS DPT_Large                     (via torch.hub)
  3. gradient_stub                        (classical fallback, no model needed)

CRITICAL SCIENTIFIC HONESTY NOTE:
  Depth Anything V2 and MiDaS are trained predominantly on ground-level,
  driving, and indoor imagery. Nadir (top-down) aerial/satellite imagery
  represents a significant domain shift. Depth output for such imagery must
  be treated as **relative structural cues only**, not metric elevation.
  This is explicitly flagged in depth_metadata.relative_only = True.
  Scale anchoring (to GSD, GeoTIFF pixel size, or reference elevation)
  is handled downstream in terrain_solver.py.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.hardware import get_hardware

log = get_logger("depth_engine")


@dataclass
class DepthMetadata:
    model_name: str
    model_backend: str            # "depth_anything_v2" | "midas" | "gradient_stub"
    device: str                   # "cuda" | "mps" | "cpu"
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    runtime_s: float
    relative_only: bool = True    # Always True unless real metric anchor applied
    domain_warning: str = (
        "Output is RELATIVE depth only. Domain shift applies to aerial/satellite "
        "imagery: models were trained on ground-level scenes. Use as structural cue, "
        "not metric elevation, unless a scale source (GSD/GeoTIFF/reference DEM) anchors it."
    )
    cache_hit: bool = False
    cache_key: str = ""


@dataclass
class DepthResult:
    depth_map: np.ndarray           # HxW float32, raw model output (upsampled to input size)
    normalized_depth: np.ndarray    # HxW float32, linearly mapped to [0, 1]
    depth_metadata: DepthMetadata


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

class DepthEngine:
    """
    Stateful depth estimation engine.
    Loads the model once and reuses it across calls.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.depth_cfg = config.get("depth", {})
        self._model = None
        self._model_name: Optional[str] = None
        hw = get_hardware()
        self.device = hw.device if hw.torch_available else "cpu"
        self._torch_available = hw.torch_available
        log.info("DepthEngine initialized | device=%s | torch=%s",
                 self.device, self._torch_available)

    def estimate(
        self,
        image_rgb: np.ndarray,
        input_hash: str,
        config_hash: str,
    ) -> DepthResult:
        """
        Run monocular depth estimation on a preprocessed RGB image.

        Args:
            image_rgb: HxWx3 uint8 RGB numpy array.
            input_hash: File hash for cache keying.
            config_hash: Config hash for cache invalidation.

        Returns:
            DepthResult with depth_map, normalized_depth, and metadata.
        """
        t0 = time.perf_counter()
        model_name = self.depth_cfg.get("model", "depth_anything_v2_small")
        cache_key = f"{input_hash}_{model_name}_{config_hash}"

        # Check cache
        if self.depth_cfg.get("cache_enabled", True):
            cached = self._load_cache(cache_key)
            if cached is not None:
                log.info("Depth cache hit: %s", cache_key)
                cached.depth_metadata.cache_hit = True
                return cached

        log.info("Running depth inference | model=%s | device=%s | image=%dx%d",
                 model_name, self.device, image_rgb.shape[1], image_rgb.shape[0])

        # Route to appropriate backend
        if not self._torch_available and model_name != "gradient_stub":
            log.warning("PyTorch not available — falling back to gradient_stub.")
            model_name = "gradient_stub"

        try:
            if model_name.startswith("depth_anything_v2"):
                depth_raw = self._run_depth_anything_v2(image_rgb, model_name)
            elif model_name.startswith("midas"):
                depth_raw = self._run_midas(image_rgb)
            else:
                depth_raw = self._run_gradient_stub(image_rgb)
        except Exception as exc:
            log.error("Depth backend '%s' failed: %s — falling back to gradient_stub.", model_name, exc)
            depth_raw = self._run_gradient_stub(image_rgb)
            model_name = "gradient_stub"

        # Upsample to original image size if needed
        h, w = image_rgb.shape[:2]
        depth_map = _resize_depth(depth_raw, (h, w))

        # Normalize to [0, 1]
        d_min, d_max = float(depth_map.min()), float(depth_map.max())
        if d_max > d_min:
            normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depth_map)
            log.warning("Depth map has no variation (min=max=%.4f). Check model/input.", d_min)

        runtime_s = time.perf_counter() - t0
        log.info("Depth inference complete | runtime=%.2fs | depth_range=[%.3f, %.3f]",
                 runtime_s, d_min, d_max)

        meta = DepthMetadata(
            model_name=model_name,
            model_backend=_get_backend_label(model_name),
            device=self.device,
            input_height=h,
            input_width=w,
            output_height=depth_raw.shape[0],
            output_width=depth_raw.shape[1],
            runtime_s=runtime_s,
            relative_only=True,
            cache_hit=False,
            cache_key=cache_key,
        )

        result = DepthResult(
            depth_map=depth_map.astype(np.float32),
            normalized_depth=normalized.astype(np.float32),
            depth_metadata=meta,
        )

        if self.depth_cfg.get("cache_enabled", True):
            self._save_cache(cache_key, result)

        return result

    # ──────────────────────────────────────────────────────────
    # Backend implementations
    # ──────────────────────────────────────────────────────────

    def _run_depth_anything_v2(self, image_rgb: np.ndarray, model_name: str) -> np.ndarray:
        """
        Run Depth Anything V2 via HuggingFace Transformers pipeline.
        Downloads from HF hub on first call, then uses local cache.
        """
        import torch  # type: ignore
        from transformers import pipeline as hf_pipeline  # type: ignore
        from PIL import Image as PILImage  # type: ignore

        # Select HF repo based on size variant
        if "large" in model_name:
            repo = self.depth_cfg.get("hf_repo_large", "depth-anything/Depth-Anything-V2-Large-hf")
        elif "base" in model_name:
            repo = self.depth_cfg.get("hf_repo_base", "depth-anything/Depth-Anything-V2-Base-hf")
        else:
            repo = self.depth_cfg.get("hf_repo_small", "depth-anything/Depth-Anything-V2-Small-hf")

        # Load model only once
        if self._model is None or self._model_name != model_name:
            log.info("Loading Depth Anything V2 from HF hub: %s", repo)
            torch_device = self.device

            # MPS dtype: float32 (not all ops support float16 on MPS yet)
            if torch_device == "mps":
                dtype = torch.float32
            elif torch_device == "cuda":
                dtype = torch.float16 if self.depth_cfg.get("use_mixed_precision", True) else torch.float32
            else:
                dtype = torch.float32

            self._model = hf_pipeline(
                task="depth-estimation",
                model=repo,
                device=torch_device,
                dtype=dtype,
            )
            self._model_name = model_name
            log.info("Depth Anything V2 loaded | device=%s | dtype=%s", torch_device, dtype)

        # Run inference
        pil_img = PILImage.fromarray(image_rgb)
        with torch.no_grad():
            result = self._model(pil_img)

        # HF depth-estimation pipeline returns dict with "predicted_depth" tensor
        depth_tensor = result["predicted_depth"]  # shape (1, H, W) or (H, W)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)
        depth_np = depth_tensor.cpu().float().numpy()
        return depth_np

    def _run_midas(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run MiDaS DPT_Large via torch.hub as a fallback."""
        import torch  # type: ignore

        model_type = self.depth_cfg.get("midas_model_type", "DPT_Large")

        if self._model is None or self._model_name != "midas":
            log.info("Loading MiDaS %s from torch.hub...", model_type)
            midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type in {"DPT_Large", "DPT_Hybrid"}:
                transform = transforms.dpt_transform
            else:
                transform = transforms.small_transform
            self._model = (midas, transform)
            self._model_name = "midas"
            midas.to(self.device)
            midas.eval()
            log.info("MiDaS %s loaded | device=%s", model_type, self.device)

        midas_model, transform = self._model
        from PIL import Image as PILImage  # type: ignore

        pil_img = PILImage.fromarray(image_rgb)
        input_batch = transform(np.array(pil_img)).to(self.device)

        with torch.no_grad():
            prediction = midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().float().numpy()

    def _run_gradient_stub(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Classical gradient-based depth approximation.
        No model required — uses image luminance gradients as a structural proxy.
        This is a transparent approximation — labeled explicitly in metadata.
        NOT intended as a real depth estimate; purely a fallback to keep the
        pipeline running when PyTorch/models are unavailable.
        """
        log.warning(
            "Using gradient_stub depth — this is a TRANSPARENT APPROXIMATION, "
            "not a real depth model. Install PyTorch + transformers for real depth."
        )
        from scipy.ndimage import gaussian_filter  # type: ignore

        gray = np.mean(image_rgb.astype(np.float32), axis=2)
        # Smooth and invert (closer = brighter in typical images)
        smoothed = gaussian_filter(gray, sigma=3.0)
        # Multi-scale gradient magnitude as a rough depth proxy
        from scipy.ndimage import sobel  # type: ignore
        gx = sobel(smoothed, axis=1)
        gy = sobel(smoothed, axis=0)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        # Depth proxy: smooth regions are "closer" (lower gradient)
        depth_proxy = gaussian_filter(255.0 - gradient_mag, sigma=10.0)
        return depth_proxy.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # Cache helpers
    # ──────────────────────────────────────────────────────────

    def _cache_path(self, cache_key: str) -> Path:
        cache_dir = Path(self.depth_cfg.get("cache_dir", "data/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}_depth.npz"

    def _save_cache(self, cache_key: str, result: DepthResult) -> None:
        try:
            p = self._cache_path(cache_key)
            np.savez_compressed(
                p,
                depth_map=result.depth_map,
                normalized_depth=result.normalized_depth,
            )
            log.debug("Depth cached: %s", p)
        except Exception as exc:
            log.warning("Could not save depth cache: %s", exc)

    def _load_cache(self, cache_key: str) -> Optional[DepthResult]:
        try:
            p = self._cache_path(cache_key)
            if not p.exists():
                return None
            data = np.load(p)
            meta = DepthMetadata(
                model_name="cached",
                model_backend="cached",
                device=self.device,
                input_height=0,
                input_width=0,
                output_height=data["depth_map"].shape[0],
                output_width=data["depth_map"].shape[1],
                runtime_s=0.0,
                relative_only=True,
                cache_hit=True,
                cache_key=cache_key,
            )
            return DepthResult(
                depth_map=data["depth_map"],
                normalized_depth=data["normalized_depth"],
                depth_metadata=meta,
            )
        except Exception as exc:
            log.warning("Could not load depth cache (will re-run inference): %s", exc)
            return None


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample/downsample depth map to (H, W)."""
    from PIL import Image as PILImage  # type: ignore

    h, w = target_hw
    if depth.shape == (h, w):
        return depth
    pil = PILImage.fromarray(depth.astype(np.float32), mode="F")
    pil = pil.resize((w, h), PILImage.BILINEAR)
    return np.array(pil, dtype=np.float32)


def _get_backend_label(model_name: str) -> str:
    if model_name.startswith("depth_anything"):
        return "depth_anything_v2"
    if model_name.startswith("midas"):
        return "midas"
    return "gradient_stub"
