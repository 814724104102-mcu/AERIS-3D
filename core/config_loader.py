"""
AERIS-3D — Configuration Loader
Loads configs/default.yaml and merges with optional override files.
All pipeline parameters must come from here — never hardcoded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.logger import configure_from_config, get_logger

log = get_logger("config")

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load AERIS-3D configuration.

    Args:
        config_path: Optional override path. Defaults to configs/default.yaml.

    Returns:
        Merged configuration dict.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the config file is malformed YAML.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    log.info("Loading configuration from: %s", path)

    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Re-configure logging using the loaded settings
    configure_from_config(cfg)

    log.debug("Config loaded: %d top-level keys", len(cfg))
    return cfg


def config_version_hash(cfg: dict[str, Any]) -> str:
    """
    Return a short hash of the config dict for use in cache keys.
    Changes to config → different hash → stale cache is invalidated.
    """
    raw = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
