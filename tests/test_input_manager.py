"""
Tests for core/input_manager.py

Covers:
- Valid JPG/PNG loading
- Unsupported format rejection
- Missing file error
- Preprocessing metadata accuracy
- GeoRef is None for RGB images (never fabricated)
- Synthetic GeoTIFF loading (generated in-memory)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config_loader import load_config
from core.input_manager import load_input, FileFormat


@pytest.fixture
def cfg():
    """Load the default config."""
    return load_config()


@pytest.fixture
def sample_jpg(tmp_path):
    """Create a temporary 256x256 RGB JPEG."""
    img = PILImage.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    p = tmp_path / "sample.jpg"
    img.save(p, format="JPEG")
    return p


@pytest.fixture
def sample_png(tmp_path):
    """Create a temporary 300x400 RGB PNG."""
    img = PILImage.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    p = tmp_path / "sample.png"
    img.save(p, format="PNG")
    return p


@pytest.fixture
def sample_large_jpg(tmp_path):
    """Create a 2048x2048 image that needs to be resized."""
    img = PILImage.fromarray(np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8))
    p = tmp_path / "large.jpg"
    img.save(p, format="JPEG")
    return p


# ──────────────────────────────────────────────────────────────
# Gate 1 — Basic loading
# ──────────────────────────────────────────────────────────────

class TestJpegLoading:
    def test_returns_input_data(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        assert result is not None

    def test_file_format_rgb(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        assert result.file_format == FileFormat.RGB_IMAGE

    def test_image_rgb_shape(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        h, w, c = result.image_rgb.shape
        assert c == 3, "image_rgb must have 3 channels"

    def test_image_rgb_dtype(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        assert result.image_rgb.dtype == np.uint8

    def test_preprocessed_dtype(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        assert result.image_preprocessed.dtype == np.float32

    def test_georef_none_for_jpeg(self, sample_jpg, cfg):
        """Critical: georef must NEVER be fabricated for RGB images."""
        result = load_input(sample_jpg, cfg)
        assert result.georef is None, "georef must be None for plain JPEG — never fabricated"

    def test_input_hash_nonempty(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        assert len(result.input_hash) > 0

    def test_preprocessing_metadata_recorded(self, sample_jpg, cfg):
        result = load_input(sample_jpg, cfg)
        meta = result.preprocessing_meta
        assert meta.original_height == 256
        assert meta.original_width == 256
        assert meta.processing_time_s >= 0.0


class TestPngLoading:
    def test_returns_correct_original_size(self, sample_png, cfg):
        result = load_input(sample_png, cfg)
        assert result.preprocessing_meta.original_height == 300
        assert result.preprocessing_meta.original_width == 400

    def test_format_rgb(self, sample_png, cfg):
        result = load_input(sample_png, cfg)
        assert result.file_format == FileFormat.RGB_IMAGE


class TestResizing:
    def test_large_image_resized(self, sample_large_jpg, cfg):
        result = load_input(sample_large_jpg, cfg)
        max_size = cfg.get("input", {}).get("max_image_size", 1024)
        ph = result.preprocessing_meta.preprocessed_height
        pw = result.preprocessing_meta.preprocessed_width
        assert max(ph, pw) <= max_size, (
            f"Preprocessed size {pw}x{ph} exceeds max_size={max_size}"
        )

    def test_small_image_not_upscaled(self, sample_jpg, cfg):
        """Images smaller than max_size should not be upscaled."""
        result = load_input(sample_jpg, cfg)
        # 256x256 << 1024 → should not be resized
        assert result.preprocessing_meta.preprocessed_height == 256
        assert result.preprocessing_meta.preprocessed_width == 256
        assert abs(result.preprocessing_meta.resize_scale - 1.0) < 1e-6


# ──────────────────────────────────────────────────────────────
# Error cases
# ──────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_file_raises(self, cfg):
        with pytest.raises(FileNotFoundError):
            load_input("/nonexistent/path/image.jpg", cfg)

    def test_unsupported_format_raises(self, tmp_path, cfg):
        p = tmp_path / "image.bmp"
        # Create a dummy file
        p.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_input(p, cfg)

    def test_corrupt_image_raises(self, tmp_path, cfg):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"not a real jpeg")
        with pytest.raises(ValueError):
            load_input(p, cfg)


# ──────────────────────────────────────────────────────────────
# GeoTIFF (synthetic)
# ──────────────────────────────────────────────────────────────

class TestGeoTIFF:
    @pytest.fixture
    def sample_geotiff(self, tmp_path):
        """Create a minimal synthetic GeoTIFF using rasterio."""
        pytest.importorskip("rasterio")
        import rasterio  # type: ignore
        from rasterio.transform import from_bounds  # type: ignore

        p = tmp_path / "sample.tif"
        h, w = 128, 128
        transform = from_bounds(77.0, 28.0, 77.1, 28.1, w, h)
        data = np.random.randint(50, 200, (3, h, w), dtype=np.uint8)

        with rasterio.open(
            p, "w",
            driver="GTiff",
            height=h, width=w,
            count=3,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
        return p

    def test_geotiff_format(self, sample_geotiff, cfg):
        result = load_input(sample_geotiff, cfg)
        assert result.file_format == FileFormat.GEOTIFF

    def test_geotiff_georef_populated(self, sample_geotiff, cfg):
        result = load_input(sample_geotiff, cfg)
        assert result.georef is not None, "GeoTIFF must have georef populated"

    def test_geotiff_crs_preserved(self, sample_geotiff, cfg):
        result = load_input(sample_geotiff, cfg)
        assert "4326" in result.georef.crs or "epsg" in result.georef.crs.lower()

    def test_geotiff_bounds_not_zero(self, sample_geotiff, cfg):
        result = load_input(sample_geotiff, cfg)
        g = result.georef
        assert g.bounds_east != g.bounds_west
        assert g.bounds_north != g.bounds_south

    def test_geotiff_image_rgb_shape(self, sample_geotiff, cfg):
        result = load_input(sample_geotiff, cfg)
        assert result.image_rgb.ndim == 3
        assert result.image_rgb.shape[2] == 3
