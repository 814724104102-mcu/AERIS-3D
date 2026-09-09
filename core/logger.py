"""
AERIS-3D — Centralized Logging
All modules import from here. Writes to both console and rotating file.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_INITIALIZED = False
_LOGGER_NAME = "aeris3d"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-specific logger under the aeris3d hierarchy."""
    global _INITIALIZED
    if not _INITIALIZED:
        _setup_root_logger()
        _INITIALIZED = True
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def _setup_root_logger(log_dir: str = "data/logs", level: str = "INFO") -> None:
    """Configure the root aeris3d logger with console + file handlers."""
    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return  # already configured

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (rotating, 10 MB × 3 backups)
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path / "aeris3d.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:
        root.warning("Could not create log file: %s — logging to console only.", exc)


def configure_from_config(cfg: dict) -> None:
    """Re-configure the root logger from a loaded config dict (call after config load)."""
    global _INITIALIZED
    log_cfg = cfg.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_dir = log_cfg.get("log_dir", "data/logs")
    _INITIALIZED = False  # force re-init with new settings
    _setup_root_logger(log_dir=log_dir, level=level)
    _INITIALIZED = True
