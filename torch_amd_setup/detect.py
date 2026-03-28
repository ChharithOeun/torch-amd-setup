"""
torch_amd_setup.detect
======================
Core device detection logic for torch-amd-setup.

Detection priority:
  1. NVIDIA CUDA          — pip install torch  (standard PyTorch)
  2. AMD ROCm (Linux)     — pip install torch --index-url .../rocm6.1
  3. AMD DirectML (Win)   — pip install torch-directml  (Python ≤3.11)
  4. Apple MPS            — built into PyTorch on macOS Apple Silicon
  5. CPU fallback         — always available, always slow

AMD gfx1010 (RX 5700 XT / Navi 10) note:
  This GPU is NOT in ROCm's default supported GPU list.
  The workaround is to set HSA_OVERRIDE_GFX_VERSION=10.3.0 before importing
  torch. This module handles that automatically.
"""

from __future__ import annotations

import os
import sys
import logging
import platform
from typing import Literal

log = logging.getLogger("torch_amd_setup")

# ── AMD gfx1010 ROCm override ─────────────────────────────────────────────────
# These must be set in the environment BEFORE torch is imported for the first
# time. get_best_device() applies them automatically when ROCm is attempted.
#
# Override map: GPU chip → HSA_OVERRIDE_GFX_VERSION value
# Add your card here if it's not in ROCm's default allow-list.
GFX_OVERRIDE_MAP: dict[str, str] = {
    "gfx1010": "10.3.0",   # RX 5700 XT, RX 5700, RX 5600 XT (Navi 10)
    "gfx1011": "10.3.0",   # RX 5500 XT (Navi 14)
    "gfx1012": "10.3.0",   # RX 5300 (Navi 14 lite)
}

# Default override applied when no specific chip is specified
AMD_ROCM_ENV: dict[str, str] = {
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "HIP_VISIBLE_DEVICES":      "0",
    "ROCR_VISIBLE_DEVICES":     "0",
}

DeviceType = Literal["cuda", "rocm", "dml", "mps", "cpu"]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL PROBES
# ─────────────────────────────────────────────────────────────────────────────

def _try_cuda() -> bool:
    """Returns True if a NVIDIA CUDA-enabled PyTorch is installed and a GPU is visible."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _try_rocm() -> bool:
    """
    Returns True if an AMD ROCm-enabled PyTorch is installed and a GPU is visible.

    Automatically applies HSA_OVERRIDE_GFX_VERSION so that GPUs not in ROCm's
    default allow-list (like gfx1010) are still usable. The env vars are set
    only if not already present, so user-set values are always respected.
    """
    for k, v in AMD_ROCM_ENV.items():
        if k not in os.environ:
            os.environ[k] = v

    try:
        import torch
        if not torch.cuda.is_available():
            return False

        dev_name = torch.cuda.get_device_name(0).lower()
        # ROCm builds show AMD device names in the CUDA compatibility layer
        if any(kw in dev_name for kw in ["amd", "radeon", "rx ", "vega", "navi"]):
            log.info("ROCm device detected: %s", dev_name)
            return True
        # Secondary check: torch ROCm builds expose torch.version.hip
        if getattr(torch.version, "hip", None):
            return True
        return False
    except (ImportError, RuntimeError):
        return False


def _try_directml():
    """
    Returns a torch_directml device object if torch-directml is installed,
    otherwise returns None.

    torch-directml requires Python ≤3.11 — silently skipped on 3.12+.
    Works on any DirectX 12 GPU on Windows (AMD, NVIDIA, Intel).
    """
    if sys.version_info >= (3, 12):
        log.debug(
            "DirectML skipped — requires Python ≤3.11 (current: %s.%s)",
            sys.version_info.major, sys.version_info.minor,
        )
        return None
    try:
        import torch_directml as dml  # type: ignore
        if dml.is_available():
            name = dml.device_name(dml.default_device())
            log.info("DirectML device ready: %s", name)
            return dml.device()
        return None
    except (ImportError, Exception):
        return None


def _try_mps() -> bool:
    """Returns True if Apple MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        return (
            platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_best_device() -> DeviceType:
    """
    Detect and return the best available compute device.

    Returns one of: 'cuda' | 'rocm' | 'dml' | 'mps' | 'cpu'

    Use get_torch_device() to convert this string into an actual
    torch.device object suitable for model.to(device).

    Priority order:
        1. NVIDIA CUDA  (Linux / Windows / WSL2)
        2. AMD ROCm     (Linux / WSL2 only)
        3. AMD DirectML (Windows, Python ≤3.11)
        4. Apple MPS    (macOS Apple Silicon)
        5. CPU fallback
    """
    # 1. NVIDIA CUDA
    if _try_cuda():
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            log.info("Device selected: CUDA — %s", name)
        except Exception:
            log.info("Device selected: CUDA")
        return "cuda"

    # 2. AMD ROCm (Linux or WSL2 with ROCm PyTorch)
    if platform.system() == "Linux" and _try_rocm():
        log.info("Device selected: AMD ROCm")
        return "rocm"

    # 3. AMD DirectML (Windows, Python ≤3.11)
    if _try_directml() is not None:
        log.info("Device selected: AMD DirectML")
        return "dml"

    # 4. Apple MPS
    if _try_mps():
        log.info("Device selected: Apple MPS")
        return "mps"

    # 5. CPU fallback
    log.warning(
        "No GPU detected — falling back to CPU. "
        "Run `python -m torch_amd_setup` for setup instructions."
    )
    return "cpu"


def get_torch_device(device_type: DeviceType | None = None):
    """
    Return a torch.device (or DirectML device object) ready for model.to().

    If device_type is None, calls get_best_device() automatically.

    Note: ROCm PyTorch exposes AMD GPUs through the CUDA compatibility layer,
    so both 'cuda' and 'rocm' map to torch.device('cuda:0').
    """
    import torch

    if device_type is None:
        device_type = get_best_device()

    if device_type in ("cuda", "rocm"):
        return torch.device("cuda:0")

    if device_type == "dml":
        try:
            import torch_directml as dml  # type: ignore
            return dml.device()
        except ImportError:
            log.warning("torch-directml not importable — falling back to CPU")
            return torch.device("cpu")

    if device_type == "mps":
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device_type: DeviceType | None = None):
    """
    Return the recommended floating-point dtype for the detected device.

    - CUDA / ROCm / MPS → torch.float16  (half-precision, saves VRAM)
    - DirectML / CPU    → torch.float32  (DirectML float16 support is unreliable)

    If device_type is None, calls get_best_device() automatically.
    """
    import torch

    if device_type is None:
        device_type = get_best_device()

    if device_type in ("cuda", "rocm", "mps"):
        return torch.float16

    return torch.float32


def device_info() -> dict:
    """
    Return a diagnostic dictionary with current GPU/device information.

    Useful for logging, bug reports, and CI environment checks.

    Returns:
        dict with keys: python_version, platform, best_device, cuda_available,
        cuda_device_name, cuda_vram_mb, rocm_available, dml_available,
        dml_device_name, mps_available, torch_version, torch_hip_version
    """
    info: dict = {
        "python_version":    sys.version[:10],
        "platform":          platform.platform(),
        "best_device":       None,
        "cuda_available":    False,
        "cuda_device_name":  None,
        "cuda_vram_mb":      None,
        "rocm_available":    False,
        "dml_available":     False,
        "dml_device_name":   None,
        "mps_available":     False,
        "torch_version":     None,
        "torch_hip_version": None,
    }

    try:
        import torch
        info["torch_version"]     = torch.__version__
        info["torch_hip_version"] = getattr(torch.version, "hip", None)
        info["cuda_available"]    = torch.cuda.is_available()

        if info["cuda_available"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            try:
                mem = torch.cuda.get_device_properties(0).total_memory
                info["cuda_vram_mb"] = round(mem / 1024 / 1024)
            except Exception:
                pass

        info["rocm_available"] = bool(getattr(torch.version, "hip", None))
        info["mps_available"]  = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    try:
        import torch_directml as dml  # type: ignore
        info["dml_available"] = dml.is_available()
        if info["dml_available"]:
            info["dml_device_name"] = dml.device_name(dml.default_device())
    except ImportError:
        pass

    info["best_device"] = get_best_device()
    return info


# ─────────────────────────────────────────────────────────────────────────────
# INSTALL GUIDES
# ─────────────────────────────────────────────────────────────────────────────

def get_install_guide() -> str:
    """Return platform-appropriate install instructions as a formatted string."""
    os_name = platform.system()
    if os_name == "Windows":
        return _GUIDE_WINDOWS_AMD
    if os_name == "Linux":
        return _GUIDE_LINUX_AMD
    if os_name == "Darwin":
        return _GUIDE_MACOS
    return _GUIDE_CPU_FALLBACK


def get_wsl2_install_guide() -> str:
    """Full step-by-step guide for AMD GPU acceleration via WSL2 + ROCm on Windows."""
    return _GUIDE_WSL2_ROCM


# ─────────────────────────────────────────────────────────────────────────────
# GUIDE TEXT
# ─────────────────────────────────────────────────────────────────────────────

_GUIDE_WINDOWS_AMD = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AMD GPU on Windows — torch-amd-setup install guide
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A — DirectML (Easiest, any DX12 GPU, Windows-native)
────────────────────────────────────────────────────────────
Requires Python 3.11 or earlier (torch-directml does not support 3.12+).

  1. Create a Python 3.11 venv:
       py -3.11 -m venv .venv311
       .venv311\\Scripts\\activate

  2. Install PyTorch (CPU wheel) + DirectML backend:
       pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
       pip install torch-directml

  3. Install torch-amd-setup:
       pip install torch-amd-setup

  4. Verify:
       python -m torch_amd_setup

  Note: DirectML only supports float32. If your model needs float16 for VRAM
        reasons, use Option B (WSL2 + ROCm) instead.

OPTION B — WSL2 + ROCm (Best quality, full float16 support)
─────────────────────────────────────────────────────────────
For step-by-step instructions run:
  python -c "from torch_amd_setup import get_wsl2_install_guide; print(get_wsl2_install_guide())"

OPTION C — CPU fallback (No setup required, very slow)
───────────────────────────────────────────────────────
  pip install torch torch-amd-setup
  # torch_amd_setup will detect CPU automatically — no extra config needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_GUIDE_WSL2_ROCM = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AMD GPU on Windows — WSL2 + ROCm setup (Best path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requirements:
  - Windows 10 21H2+ or Windows 11
  - AMD Adrenalin driver 22.20 or newer
  - AMD GPU with ROCm support (or gfx1010 override for RX 5700 XT)

STEP 1 — Install WSL2 + Ubuntu 22.04
──────────────────────────────────────
  # In PowerShell (run as Administrator):
  wsl --install -d Ubuntu-22.04
  # Reboot when prompted, complete Ubuntu first-run setup.

STEP 2 — Remove Ubuntu's conflicting ROCm packages
─────────────────────────────────────────────────────
  # Ubuntu 22.04 ships rocminfo 5.0.0 which blocks ROCm 6.1:
  sudo apt-get remove -y rocminfo rocm-device-libs rocm-cmake rocm-hip-sdk rocm-hip-runtime 2>/dev/null || true

STEP 3 — Install ROCm 6.1
───────────────────────────
  wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \\
      gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
  printf 'Package: *\\nPin: origin repo.radeon.com\\nPin-Priority: 1001\\n' | \\
      sudo tee /etc/apt/preferences.d/rocm-pin
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \\
      https://repo.radeon.com/rocm/apt/6.1 jammy main" | \\
      sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update && sudo apt install -y rocm-hip-sdk python3-dev python3-pip
  sudo usermod -aG render,video $USER && newgrp render

STEP 4 — Apply gfx1010 override (RX 5700 XT / Navi 10 only)
──────────────────────────────────────────────────────────────
  echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
  echo 'export HIP_VISIBLE_DEVICES=0'           >> ~/.bashrc
  source ~/.bashrc

STEP 5 — Install PyTorch ROCm build
─────────────────────────────────────
  python3 -m venv ~/.venvs/rocm
  source ~/.venvs/rocm/bin/activate
  pip install torch torchvision torchaudio \\
      --index-url https://download.pytorch.org/whl/rocm6.1
  pip install torch-amd-setup

STEP 6 — Verify
────────────────
  python3 -m torch_amd_setup
  # Expected: Device: rocm | AMD Radeon RX 5700 XT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_GUIDE_LINUX_AMD = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AMD GPU on Linux — ROCm setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Install ROCm 6.1 (Ubuntu 22.04):
       wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \\
           gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
       echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \\
           https://repo.radeon.com/rocm/apt/6.1 jammy main" | \\
           sudo tee /etc/apt/sources.list.d/rocm.list
       sudo apt update && sudo apt install -y rocm-hip-sdk
       sudo usermod -aG render,video $USER && newgrp render

  2. (RX 5700 XT only) Add to ~/.bashrc:
       export HSA_OVERRIDE_GFX_VERSION=10.3.0
       export HIP_VISIBLE_DEVICES=0

  3. Install PyTorch ROCm + torch-amd-setup:
       pip install torch torchvision torchaudio \\
           --index-url https://download.pytorch.org/whl/rocm6.1
       pip install torch-amd-setup

  4. Verify:
       python3 -m torch_amd_setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_GUIDE_MACOS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 macOS Apple Silicon — MPS setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install torch torchvision torchaudio torch-amd-setup
  # MPS is auto-detected — no extra configuration needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_GUIDE_CPU_FALLBACK = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CPU-only mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install torch torch-amd-setup
  # CPU is selected automatically when no GPU is detected.

  WARNING: CPU inference is 10–30x slower than real-time for large models.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI SELF-TEST  (python -m torch_amd_setup)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    print("\n── torch-amd-setup diagnostics ──────────────────────────────")
    info = device_info()
    for k, v in info.items():
        print(f"  {k:<25} {v}")
    print()
    print("── Install guide ─────────────────────────────────────────────")
    print(get_install_guide())
