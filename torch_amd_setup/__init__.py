"""
torch-amd-setup
~~~~~~~~~~~~~~~
Auto-detects the best PyTorch compute device across AMD ROCm, AMD DirectML,
NVIDIA CUDA, Apple MPS, and CPU — with first-class support for AMD cards that
are not in ROCm's default allow-list (e.g. RX 5700 XT / gfx1010).

Basic usage:
    from torch_amd_setup import get_best_device, get_torch_device, get_dtype

    device_type = get_best_device()       # "cuda" | "rocm" | "dml" | "mps" | "cpu"
    device      = get_torch_device()      # torch.device (or DirectML device)
    dtype       = get_dtype()             # torch.float16 or torch.float32
"""

from .detect import (
    get_best_device,
    get_torch_device,
    get_dtype,
    device_info,
    get_install_guide,
    get_wsl2_install_guide,
    AMD_ROCM_ENV,
    DeviceType,
)

__version__ = "0.1.0"
__all__ = [
    "get_best_device",
    "get_torch_device",
    "get_dtype",
    "device_info",
    "get_install_guide",
    "get_wsl2_install_guide",
    "AMD_ROCM_ENV",
    "DeviceType",
]
