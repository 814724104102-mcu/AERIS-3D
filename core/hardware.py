"""
AERIS-3D — Hardware Detection
Auto-detects the best compute backend: CUDA → MPS → CPU.
Never crashes if GPU libraries are unavailable.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Literal

from core.logger import get_logger

log = get_logger("hardware")

DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class HardwareInfo:
    device: DeviceType
    device_name: str
    torch_available: bool
    cuda_available: bool
    mps_available: bool
    cpu_count: int
    platform: str
    python_version: str
    torch_version: str


def detect_hardware() -> HardwareInfo:
    """
    Probe the runtime environment and return the best available compute device.

    Priority: CUDA > MPS > CPU
    Never raises — always returns a valid HardwareInfo.
    """
    cuda_available = False
    mps_available = False
    torch_available = False
    torch_version = "not_installed"
    device: DeviceType = "cpu"
    device_name = "CPU"

    try:
        import torch  # type: ignore

        torch_available = True
        torch_version = torch.__version__

        cuda_available = torch.cuda.is_available()
        try:
            mps_available = torch.backends.mps.is_available()
        except AttributeError:
            mps_available = False

        if cuda_available:
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif mps_available:
            device = "mps"
            device_name = "Apple MPS (Metal Performance Shaders)"
        else:
            device = "cpu"
            device_name = "CPU"

    except ImportError:
        log.warning("PyTorch not installed — running on CPU-only mode. "
                    "Install torch for GPU acceleration.")

    info = HardwareInfo(
        device=device,
        device_name=device_name,
        torch_available=torch_available,
        cuda_available=cuda_available,
        mps_available=mps_available,
        cpu_count=os.cpu_count() or 1,
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch_version,
    )

    log.info(
        "Hardware detected | device=%s (%s) | CUDA=%s | MPS=%s | CPUs=%d | torch=%s",
        info.device,
        info.device_name,
        info.cuda_available,
        info.mps_available,
        info.cpu_count,
        info.torch_version,
    )
    return info


# Module-level singleton — detected once at import time (or lazily)
_hw_info: HardwareInfo | None = None


def get_hardware() -> HardwareInfo:
    """Return cached hardware info (lazy singleton)."""
    global _hw_info
    if _hw_info is None:
        _hw_info = detect_hardware()
    return _hw_info
