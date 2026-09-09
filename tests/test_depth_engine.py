"""
Tests for core/depth_engine.py

Covers:
- DepthResult shape and dtype
- normalized_depth in [0, 1]
- relative_only always True
- depth_metadata fields present
- Cache save/load round-trip
- gradient_stub fallback
- Device detection integration
- Output shape matches input shape
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config_loader import load_config
from core.depth_engine import DepthEngine


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def stub_cfg(tmp_path):
    """Config that forces gradient_stub — no model download needed."""
    cfg = load_config()
    cfg.setdefault("depth", {})["model"] = "gradient_stub"
    cfg["depth"]["cache_dir"] = str(tmp_path / "cache")
    cfg["depth"]["cache_enabled"] = True
    return cfg


@pytest.fixture
def sample_image():
    """256x256 synthetic RGB image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """64x64 synthetic image for fast tests."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
# gradient_stub backend (no model download)
# ──────────────────────────────────────────────────────────────


class TestGradientStub:
    def test_returns_depth_result(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_test", "cfg_v0")
        assert result is not None

    def test_depth_map_dtype(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_dtype", "cfg_v0")
        assert result.depth_map.dtype == np.float32

    def test_depth_map_shape_matches_input(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_shape", "cfg_v0")
        assert (
            result.depth_map.shape == sample_image.shape[:2]
        ), f"Expected {sample_image.shape[:2]}, got {result.depth_map.shape}"

    def test_normalized_depth_in_0_1(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_norm", "cfg_v0")
        nd = result.normalized_depth
        assert (
            nd.min() >= -0.001 and nd.max() <= 1.001
        ), f"normalized_depth out of range: [{nd.min():.4f}, {nd.max():.4f}]"

    def test_normalized_depth_dtype(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_nd_dtype", "cfg_v0")
        assert result.normalized_depth.dtype == np.float32

    def test_normalized_depth_shape_matches_input(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_nd_shape", "cfg_v0")
        assert result.normalized_depth.shape == sample_image.shape[:2]

    def test_relative_only_always_true(self, stub_cfg, sample_image):
        """relative_only must always be True — it's never overridden by the engine itself."""
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_rel", "cfg_v0")
        assert (
            result.depth_metadata.relative_only is True
        ), "relative_only must be True — metric anchoring is the terrain_solver's job"

    def test_metadata_model_name(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_meta", "cfg_v0")
        assert result.depth_metadata.model_name == "gradient_stub"

    def test_metadata_runtime_recorded(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_rt", "cfg_v0")
        assert result.depth_metadata.runtime_s >= 0.0

    def test_metadata_device_nonempty(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_dev", "cfg_v0")
        assert result.depth_metadata.device in {"cuda", "mps", "cpu"}

    def test_domain_warning_present(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_dw", "cfg_v0")
        assert len(result.depth_metadata.domain_warning) > 10


# ──────────────────────────────────────────────────────────────
# Cache behaviour
# ──────────────────────────────────────────────────────────────


class TestCaching:
    def test_cache_miss_on_first_call(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(sample_image, "hash_c1", "cfg_c1")
        assert result.depth_metadata.cache_hit is False

    def test_cache_hit_on_second_call(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        engine.estimate(sample_image, "hash_c2", "cfg_c2")
        result2 = engine.estimate(sample_image, "hash_c2", "cfg_c2")
        assert result2.depth_metadata.cache_hit is True

    def test_different_hash_no_cache_hit(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        engine.estimate(sample_image, "hash_a", "cfg_v")
        result = engine.estimate(sample_image, "hash_b", "cfg_v")
        assert result.depth_metadata.cache_hit is False

    def test_cached_depth_values_match(self, stub_cfg, sample_image):
        engine = DepthEngine(stub_cfg)
        r1 = engine.estimate(sample_image, "hash_cv", "cfg_cv")
        r2 = engine.estimate(sample_image, "hash_cv", "cfg_cv")
        np.testing.assert_array_almost_equal(
            r1.depth_map,
            r2.depth_map,
            decimal=5,
            err_msg="Cached depth map differs from original",
        )

    def test_cache_disabled(self, tmp_path, sample_image):
        cfg = load_config()
        cfg.setdefault("depth", {})["model"] = "gradient_stub"
        cfg["depth"]["cache_enabled"] = False
        cfg["depth"]["cache_dir"] = str(tmp_path / "cache")
        engine = DepthEngine(cfg)
        engine.estimate(sample_image, "hash_nc", "cfg_nc")
        result = engine.estimate(sample_image, "hash_nc", "cfg_nc")
        # With caching disabled, every call is a cache miss
        assert result.depth_metadata.cache_hit is False


# ──────────────────────────────────────────────────────────────
# Non-square images
# ──────────────────────────────────────────────────────────────


class TestNonSquareImages:
    @pytest.mark.parametrize("shape", [(128, 256), (300, 100), (64, 512)])
    def test_output_shape_matches_input(self, stub_cfg, shape):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
        engine = DepthEngine(stub_cfg)
        result = engine.estimate(img, f"hash_{shape[0]}x{shape[1]}", "cfg_ns")
        assert (
            result.depth_map.shape == shape
        ), f"Expected {shape}, got {result.depth_map.shape}"
