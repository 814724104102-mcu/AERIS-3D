"""
Tests for core/pipeline.py (Phases 3–12)
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from core.config_loader import load_config
from core.depth_engine import DepthEngine
from core.input_manager import load_input
from core.pipeline import run_pipeline


@pytest.fixture
def cfg():
    config = load_config()
    config.setdefault("depth", {})["model"] = "gradient_stub"
    return config


@pytest.fixture
def sample_input(cfg):
    img_path = Path("data/input/synthetic_test.jpg")
    return load_input(img_path, cfg)


class TestPipeline:
    def test_pipeline_execution(self, cfg, sample_input):
        engine = DepthEngine(cfg)
        depth_res = engine.estimate(sample_input.image_rgb, "hash_p", "cfg_p")

        result = run_pipeline(sample_input, depth_res, cfg)

        assert result is not None
        assert result.n_candidates > 0
        assert result.n_survived > 0
        assert len(result.sdrl_result.survivors) == result.n_survived
        assert len(result.chf_result.height_fingerprint) == result.n_survived
        assert result.total_runtime_s >= 0.0
