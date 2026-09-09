"""
pytest conftest.py — shared fixtures and configuration for AERIS-3D tests.
"""
import sys
from pathlib import Path

# Ensure repo root is always on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
