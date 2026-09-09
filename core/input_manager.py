"""
AERIS-3D — Input Manager (Phase 1)
Accepts JPG/JPEG/PNG/GeoTIFF and returns a unified InputData object.

Key rules:
- Never fabricate georeferencing — if metadata is absent, georef is None.
- Returns the original image in RGB uint8 and a preprocessed float32 version.
- GeoTIFF: preserves CRS, affine transform, bounds, pixel size.
- All preprocessing is recorded in metadata for traceability.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from core.logger import get_logger

log = get_logger("input_manager")


class FileFormat(str, Enum):
    RGB_IMAGE = "RGB_IMAGE"  # standard JPG/PNG without georeferencing
    GEOTIFF = "GEOTIFF"  # raster with CRS / affine / bounds metadata


@dataclass
class GeoRef:
    """Georeferencing metadata, sourced directly from file — never fabricated."""

    crs: str  # e.g. "EPSG:4326"
    affine_transform: list[float]  # 6-element Affine coefficients [a,b,c,d,e,f]
    bounds_west: float
    bounds_south: float
    bounds_east: float
    bounds_north: float
    pixel_size_x: float  # metres/px or degrees/px
    pixel_size_y: float
    num_bands: int
    nodata_value: Optional[float]


@dataclass
class PreprocessingMetadata:
    original_height: int
    original_width: int
    original_channels: int
    preprocessed_height: int
    preprocessed_width: int
    resize_scale: float  # longest-edge scale factor applied
    normalize_mean: list[float]
    normalize_std: list[float]
    processing_time_s: float


@dataclass
class InputData:
    """
    Unified input representation for the AERIS-3D pipeline.
    All downstream modules receive this object — not raw files.
    """

    file_path: str
    file_format: FileFormat
    input_hash: str  # SHA256 of file bytes — used for caching

    # Original RGB image (HxWx3, uint8, values 0-255)
    image_rgb: np.ndarray

    # Preprocessed image (HxWx3, float32, values roughly 0-1 after normalization)
    image_preprocessed: np.ndarray

    preprocessing_meta: PreprocessingMetadata

    # Only populated for GeoTIFF inputs — NEVER fabricated for RGB images
    georef: Optional[GeoRef] = None

    # Extra raster metadata for GeoTIFF (driver, dtype, count, etc.)
    raster_meta: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def load_input(
    file_path: str | Path,
    config: dict,
) -> InputData:
    """
    Load a single image file and return a fully populated InputData.

    Args:
        file_path: Absolute or relative path to JPG/PNG/GeoTIFF.
        config: Loaded AERIS-3D config dict (from config_loader.load_config).

    Returns:
        InputData with image, preprocessing metadata, and optional georef.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Unsupported format or corrupt/unreadable file.
    """
    t0 = time.perf_counter()
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower().lstrip(".")
    supported = set(
        config.get("input", {}).get(
            "supported_formats", ["jpg", "jpeg", "png", "tif", "tiff"]
        )
    )
    if suffix not in supported:
        raise ValueError(
            f"Unsupported file format '.{suffix}'. " f"Supported: {sorted(supported)}"
        )

    log.info("Loading input: %s (format: .%s)", path.name, suffix)

    # Compute file hash for caching (hash first 64 KB + file size for speed)
    input_hash = _compute_file_hash(path)
    log.debug("Input hash: %s", input_hash)

    max_size = config.get("input", {}).get("max_image_size", 1024)
    mean = config.get("input", {}).get("normalize_mean", [0.485, 0.456, 0.406])
    std = config.get("input", {}).get("normalize_std", [0.229, 0.224, 0.225])

    if suffix in {"tif", "tiff"}:
        result = _load_geotiff(path, max_size, mean, std, input_hash, t0)
    else:
        result = _load_rgb_image(path, max_size, mean, std, input_hash, t0)

    elapsed = time.perf_counter() - t0
    log.info(
        "Input loaded | format=%s | original=%dx%d | preprocessed=%dx%d | georef=%s | %.2fs",
        result.file_format.value,
        result.preprocessing_meta.original_height,
        result.preprocessing_meta.original_width,
        result.preprocessing_meta.preprocessed_height,
        result.preprocessing_meta.preprocessed_width,
        "YES" if result.georef else "NO",
        elapsed,
    )
    return result


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────


def _compute_file_hash(path: Path) -> str:
    """SHA256 of file — uses first 64 KB + total size for speed on large rasters."""
    hasher = hashlib.sha256()
    file_size = path.stat().st_size
    with open(path, "rb") as fh:
        chunk = fh.read(65536)
        hasher.update(chunk)
    hasher.update(str(file_size).encode())
    return hasher.hexdigest()[:16]


def _resize_image(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    """
    Resize image so the longest edge ≤ max_size, preserving aspect ratio.
    Returns (resized_image, scale_factor).
    """
    from PIL import Image as PILImage  # type: ignore

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image, 1.0

    scale = max_size / longest
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil_img = PILImage.fromarray(image)
    pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil_img), scale


def _normalize(
    image_uint8: np.ndarray, mean: list[float], std: list[float]
) -> np.ndarray:
    """Normalize uint8 RGB [0,255] → float32 using per-channel mean/std."""
    img_f32 = image_uint8.astype(np.float32) / 255.0
    m = np.array(mean, dtype=np.float32)
    s = np.array(std, dtype=np.float32)
    return (img_f32 - m) / s


def _load_rgb_image(
    path: Path,
    max_size: int,
    mean: list[float],
    std: list[float],
    input_hash: str,
    t0: float,
) -> InputData:
    """Load a standard JPG/PNG file."""
    try:
        from PIL import Image as PILImage  # type: ignore
    except ImportError as exc:
        raise ImportError("Pillow is required. Install: pip install Pillow") from exc

    try:
        pil_img = PILImage.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot read image file '{path}': {exc}") from exc

    original_rgb = np.array(pil_img, dtype=np.uint8)
    orig_h, orig_w = original_rgb.shape[:2]

    resized_rgb, scale = _resize_image(original_rgb, max_size)
    pre_h, pre_w = resized_rgb.shape[:2]

    preprocessed = _normalize(resized_rgb, mean, std)
    elapsed = time.perf_counter() - t0

    return InputData(
        file_path=str(path),
        file_format=FileFormat.RGB_IMAGE,
        input_hash=input_hash,
        image_rgb=original_rgb,
        image_preprocessed=preprocessed,
        georef=None,  # never fabricated
        raster_meta={},
        preprocessing_meta=PreprocessingMetadata(
            original_height=orig_h,
            original_width=orig_w,
            original_channels=3,
            preprocessed_height=pre_h,
            preprocessed_width=pre_w,
            resize_scale=scale,
            normalize_mean=mean,
            normalize_std=std,
            processing_time_s=elapsed,
        ),
    )


def _load_geotiff(
    path: Path,
    max_size: int,
    mean: list[float],
    std: list[float],
    input_hash: str,
    t0: float,
) -> InputData:
    """
    Load a GeoTIFF, preserving all available georeferencing metadata.
    Supports single-band (greyscale/elevation) and multi-band (RGB) rasters.
    """
    try:
        import rasterio  # type: ignore
        from rasterio.enums import ColorInterp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF support. Install: pip install rasterio"
        ) from exc

    try:
        with rasterio.open(path) as src:
            crs_str = str(src.crs) if src.crs else "UNKNOWN_CRS"
            transform = src.transform
            bounds = src.bounds
            num_bands = src.count
            nodata = src.nodata
            raster_meta = {
                "driver": src.driver,
                "dtype": str(src.dtypes[0]),
                "count": src.count,
                "width": src.width,
                "height": src.height,
                "crs": crs_str,
            }

            # Read into numpy — handle 1-band, 3-band, and 4-band
            data = src.read()  # shape: (bands, H, W)

        # Convert to HxWx3 RGB uint8
        if num_bands >= 3:
            # Use first 3 bands as RGB
            rgb = np.stack([data[0], data[1], data[2]], axis=-1)
        elif num_bands == 1:
            # Greyscale / elevation band → replicate to 3 channels
            band = data[0]
            # Normalize to 0-255 for display
            b_min, b_max = band.min(), band.max()
            if b_max > b_min:
                band_norm = ((band - b_min) / (b_max - b_min) * 255).astype(np.uint8)
            else:
                band_norm = np.zeros_like(band, dtype=np.uint8)
            rgb = np.stack([band_norm, band_norm, band_norm], axis=-1)
        else:
            raise ValueError(f"Unsupported band count {num_bands} in {path}")

        # Clip to valid uint8 range
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        orig_h, orig_w = rgb.shape[:2]

        # Affine transform: 6 coefficients
        affine = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]

        georef = GeoRef(
            crs=crs_str,
            affine_transform=affine,
            bounds_west=bounds.left,
            bounds_south=bounds.bottom,
            bounds_east=bounds.right,
            bounds_north=bounds.top,
            pixel_size_x=abs(transform.a),
            pixel_size_y=abs(transform.e),
            num_bands=num_bands,
            nodata_value=nodata,
        )
        log.info(
            "GeoTIFF georef | CRS=%s | bounds=(%.4f,%.4f,%.4f,%.4f) | px_size=(%.4f,%.4f)",
            crs_str,
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            abs(transform.a),
            abs(transform.e),
        )

    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"Cannot open GeoTIFF '{path}': {exc}") from exc

    resized_rgb, scale = _resize_image(rgb, max_size)
    pre_h, pre_w = resized_rgb.shape[:2]
    preprocessed = _normalize(resized_rgb, mean, std)
    elapsed = time.perf_counter() - t0

    return InputData(
        file_path=str(path),
        file_format=FileFormat.GEOTIFF,
        input_hash=input_hash,
        image_rgb=rgb,
        image_preprocessed=preprocessed,
        georef=georef,
        raster_meta=raster_meta,
        preprocessing_meta=PreprocessingMetadata(
            original_height=orig_h,
            original_width=orig_w,
            original_channels=num_bands,
            preprocessed_height=pre_h,
            preprocessed_width=pre_w,
            resize_scale=scale,
            normalize_mean=mean,
            normalize_std=std,
            processing_time_s=elapsed,
        ),
    )
