"""
Tests for Phases 3-12 core modules.

Runs fast: uses small synthetic images + gradient_stub depth.
No model downloads. Exercises all major module contracts.
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
from core.input_manager import load_input

# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def stub_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cache")
    cfg = load_config()
    cfg.setdefault("depth", {})["model"] = "gradient_stub"
    cfg["depth"]["cache_dir"] = str(tmp)
    cfg["depth"]["cache_enabled"] = False
    return cfg


@pytest.fixture(scope="module")
def sample_image_rgb():
    """128x128 synthetic image with blocks (simulates buildings)."""
    rng = np.random.default_rng(7)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = [50, 60, 80]
    for _ in range(5):
        bx = rng.integers(10, 90)
        by = rng.integers(10, 90)
        bw = rng.integers(15, 40)
        bh = rng.integers(15, 40)
        img[by : by + bh, bx : bx + bw] = rng.integers(100, 200, 3).tolist()
    # Road stripes
    img[60:68, :] = [60, 60, 60]
    img[:, 60:68] = [60, 60, 60]
    return img


@pytest.fixture(scope="module")
def depth_map(sample_image_rgb, stub_cfg):
    engine = DepthEngine(stub_cfg)
    result = engine.estimate(sample_image_rgb, "test_hash", "test_cfg")
    return result.depth_map


# ─────────────────────────────────────────────────────────────
# Phase 3 — Structure Engine
# ─────────────────────────────────────────────────────────────


class TestStructureEngine:
    def test_returns_result(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert result is not None

    def test_edge_map_shape(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert result.edge_map.shape == sample_image_rgb.shape[:2]

    def test_label_map_shape(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert result.label_map.shape == sample_image_rgb.shape[:2]

    def test_segmentation_quality_in_range(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert 0.0 <= result.segmentation_quality <= 1.0

    def test_terrain_depth_level_is_float(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert isinstance(result.terrain_depth_level, float)

    def test_runtime_recorded(self, sample_image_rgb, depth_map, stub_cfg):
        from core.structure_engine import StructureEngine

        se = StructureEngine(stub_cfg)
        result = se.analyze(sample_image_rgb, depth_map)
        assert result.runtime_s >= 0.0

    def test_graceful_on_uniform_image(self, stub_cfg):
        """Should not raise on a completely uniform image."""
        from core.structure_engine import StructureEngine

        uniform = np.full((64, 64, 3), 128, dtype=np.uint8)
        depth = np.ones((64, 64), dtype=np.float32)
        se = StructureEngine(stub_cfg)
        result = se.analyze(uniform, depth)
        assert result is not None


# ─────────────────────────────────────────────────────────────
# Phase 4 — HCDC
# ─────────────────────────────────────────────────────────────


class TestHCDC:
    def test_corrected_depth_shape(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        result = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        assert result.corrected_depth.shape == depth_map.shape

    def test_corrected_depth_dtype(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        result = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        assert result.corrected_depth.dtype == np.float32

    def test_correction_map_nonnegative(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        result = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        assert result.correction_map.min() >= 0.0

    def test_flagged_fraction_in_range(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        result = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        assert 0.0 <= result.flagged_fraction <= 1.0


# ─────────────────────────────────────────────────────────────
# Phase 10 — Terrain Solver
# ─────────────────────────────────────────────────────────────


class TestTerrainSolver:
    def test_terrain_surface_shape(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine
        from core.terrain_solver import solve_terrain

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        hcdc = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        result = solve_terrain(hcdc.corrected_depth, sr, None, stub_cfg)
        assert result.terrain_surface.shape == depth_map.shape

    def test_scale_mode_relative_without_georef(
        self, sample_image_rgb, depth_map, stub_cfg
    ):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine
        from core.terrain_solver import RELATIVE_LABEL, solve_terrain

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        hcdc = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        result = solve_terrain(hcdc.corrected_depth, sr, None, stub_cfg)
        assert result.scale_mode == RELATIVE_LABEL

    def test_object_height_map_nonnegative(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine
        from core.terrain_solver import solve_terrain

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        hcdc = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        result = solve_terrain(hcdc.corrected_depth, sr, None, stub_cfg)
        assert result.object_height_map.min() >= 0.0


# ─────────────────────────────────────────────────────────────
# Phase 5 — Candidate Generator
# ─────────────────────────────────────────────────────────────


class TestCandidateGenerator:
    @pytest.fixture(scope="class")
    def structure_and_terrain(self, sample_image_rgb, depth_map, stub_cfg):
        from core.hcdc import apply_hcdc
        from core.structure_engine import StructureEngine
        from core.terrain_solver import solve_terrain

        sr = StructureEngine(stub_cfg).analyze(sample_image_rgb, depth_map)
        hcdc = apply_hcdc(depth_map, sample_image_rgb, sr, stub_cfg)
        tr = solve_terrain(hcdc.corrected_depth, sr, None, stub_cfg)
        return sr, tr, hcdc.corrected_depth

    def test_candidates_nonempty_when_buildings_exist(
        self, structure_and_terrain, stub_cfg
    ):
        from core.candidate_generator import generate_candidates

        sr, tr, cd = structure_and_terrain
        if not sr.buildings:
            pytest.skip("No buildings in this synthetic image")
        cands = generate_candidates(sr, tr, cd, stub_cfg)
        assert len(cands) > 0

    def test_each_candidate_has_height(self, structure_and_terrain, stub_cfg):
        from core.candidate_generator import generate_candidates

        sr, tr, cd = structure_and_terrain
        if not sr.buildings:
            pytest.skip("No buildings")
        cands = generate_candidates(sr, tr, cd, stub_cfg)
        for c in cands:
            assert c.height_value > 0

    def test_each_candidate_has_object_id(self, structure_and_terrain, stub_cfg):
        from core.candidate_generator import generate_candidates

        sr, tr, cd = structure_and_terrain
        if not sr.buildings:
            pytest.skip("No buildings")
        cands = generate_candidates(sr, tr, cd, stub_cfg)
        for c in cands:
            assert c.object_id != ""

    def test_candidate_height_unit_is_relative(self, structure_and_terrain, stub_cfg):
        from core.candidate_generator import generate_candidates

        sr, tr, cd = structure_and_terrain
        if not sr.buildings:
            pytest.skip("No buildings")
        cands = generate_candidates(sr, tr, cd, stub_cfg)
        for c in cands:
            assert "RELATIVE" in c.height_unit


# ─────────────────────────────────────────────────────────────
# Phase 11 + 12 — Scoring + SDRL
# ─────────────────────────────────────────────────────────────


class TestPipelineEndToEnd:
    def test_pipeline_runs_without_error(
        self, sample_image_rgb, depth_map, stub_cfg, tmp_path
    ):
        """Integration test: full pipeline from depth through SDRL."""
        from core.depth_engine import DepthEngine
        from core.input_manager import load_input
        from core.pipeline import run_pipeline

        # Create a temp image file
        img_path = tmp_path / "test.jpg"
        PILImage.fromarray(sample_image_rgb).save(img_path)
        input_data = load_input(img_path, stub_cfg)
        engine = DepthEngine(stub_cfg)
        dr = engine.estimate(sample_image_rgb, "pipe_hash", "pipe_cfg")

        result = run_pipeline(input_data, dr, stub_cfg)
        assert result is not None

    def test_sdrl_produces_at_least_one_survivor(
        self, sample_image_rgb, depth_map, stub_cfg, tmp_path
    ):
        from core.depth_engine import DepthEngine
        from core.input_manager import load_input
        from core.pipeline import run_pipeline

        img_path = tmp_path / "test2.jpg"
        PILImage.fromarray(sample_image_rgb).save(img_path)
        input_data = load_input(img_path, stub_cfg)
        engine = DepthEngine(stub_cfg)
        dr = engine.estimate(sample_image_rgb, "pipe_hash2", "pipe_cfg2")

        result = run_pipeline(input_data, dr, stub_cfg)
        if result.candidates:
            assert (
                len(result.sdrl_result.survivors) >= 1
            ), "SDRL must always produce at least one survivor"

    def test_sdrl_log_has_entries(
        self, sample_image_rgb, depth_map, stub_cfg, tmp_path
    ):
        from core.depth_engine import DepthEngine
        from core.input_manager import load_input
        from core.pipeline import run_pipeline

        img_path = tmp_path / "test3.jpg"
        PILImage.fromarray(sample_image_rgb).save(img_path)
        input_data = load_input(img_path, stub_cfg)
        engine = DepthEngine(stub_cfg)
        dr = engine.estimate(sample_image_rgb, "pipe_hash3", "pipe_cfg3")

        result = run_pipeline(input_data, dr, stub_cfg)
        if result.candidates:
            assert len(result.sdrl_result.iteration_log) > 0

    def test_relative_only_propagated(
        self, sample_image_rgb, depth_map, stub_cfg, tmp_path
    ):
        from core.depth_engine import DepthEngine
        from core.input_manager import load_input
        from core.pipeline import run_pipeline

        img_path = tmp_path / "test4.jpg"
        PILImage.fromarray(sample_image_rgb).save(img_path)
        input_data = load_input(img_path, stub_cfg)
        engine = DepthEngine(stub_cfg)
        dr = engine.estimate(sample_image_rgb, "pipe_hash4", "pipe_cfg4")

        result = run_pipeline(input_data, dr, stub_cfg)
        assert result.depth_result.depth_metadata.relative_only is True

    def test_scale_mode_relative_without_georef(
        self, sample_image_rgb, depth_map, stub_cfg, tmp_path
    ):
        from core.depth_engine import DepthEngine
        from core.input_manager import load_input
        from core.pipeline import run_pipeline

        img_path = tmp_path / "test5.jpg"
        PILImage.fromarray(sample_image_rgb).save(img_path)
        input_data = load_input(img_path, stub_cfg)
        engine = DepthEngine(stub_cfg)
        dr = engine.estimate(sample_image_rgb, "pipe_hash5", "pipe_cfg5")

        result = run_pipeline(input_data, dr, stub_cfg)
        assert result.scale_mode == "RELATIVE"
