#!/usr/bin/env python3
"""
setup_env.py — Universal cross-platform setup for torch-amd-setup
==================================================================

Works on Windows, Linux, macOS — no bash, no batch files needed.
Uses only Python standard library + subprocess.

Usage:
    python setup_env.py                  # auto-detect and install
    python setup_env.py --check          # verify environment
    python setup_env.py --directml       # Windows AMD GPU (Python 3.11 req)
    python setup_env.py --rocm           # Linux/WSL2 AMD GPU
    python setup_env.py --cuda           # NVIDIA GPU
    python setup_env.py --mps            # macOS Apple Silicon
    python setup_env.py --cpu            # CPU only (any OS)

Python: 3.8+ for this script. DirectML: Python 3.11 only (ABI ceiling).
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

OS = platform.system()
PYTHON = sys.executable
PY_VER = sys.version_info[:2]


def banner(msg: str):
    print("=" * 66)
    print(f"  {msg}")
    print("=" * 66)


def run(cmd: list[str]):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def check_env():
    banner("Environment Check — torch-amd-setup")
    print(f"  Python      : {sys.version}")
    print(f"  OS          : {OS} {platform.version()}")
    print(f"  Machine     : {platform.machine()}")
    print()

    # torch
    try:
        import torch
        print(f"  torch       : OK  {torch.__version__}")
        cuda = torch.cuda.is_available()
        if cuda:
            hip = getattr(torch.version, "hip", None)
            label = "ROCm" if hip else "CUDA"
            print(f"  {label:<12}: OK  {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA/ROCm   : not available")
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"  MPS (Apple) : {'OK' if mps else 'not available'}")
    except ImportError:
        print("  torch       : NOT INSTALLED")

    if OS == "Windows":
        try:
            import torch_directml as dml
            gname = dml.device_name(0) if hasattr(dml, "device_name") else "unknown"
            print(f"  DirectML    : OK  {dml.__version__}  ({gname})")
        except ImportError:
            print("  DirectML    : not installed  (pip install torch-directml, Python 3.11 only)")
    else:
        print(f"  DirectML    : N/A (Windows only)")

    # torch-amd-setup
    try:
        from torch_amd_setup import get_best_device, device_info
        best = get_best_device()
        print(f"\n  amd-setup device: {best}")
        info = device_info()
        for k, v in list(info.items())[:6]:
            print(f"    {k:<25}: {v}")
    except ImportError:
        print("\n  torch-amd-setup : NOT INSTALLED  (pip install torch-amd-setup)")

    print("=" * 66)


def _detect_best() -> str:
    if OS == "Windows":
        try:
            import torch_directml  # noqa: F401
            return "directml"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _install_torch_amd_setup():
    run([PYTHON, "-m", "pip", "install", "torch-amd-setup"])


def install_directml():
    banner("Installing torch-directml + torch-amd-setup (Windows, Python 3.11)")
    if PY_VER > (3, 11):
        print(f"[ERROR] Python {PY_VER[0]}.{PY_VER[1]} — DirectML requires Python <= 3.11.")
        print("  Create a 3.11 venv:  py -3.11 -m venv .venv311")
        print("  Activate:            .venv311\\Scripts\\activate")
        print("  Then re-run:         python setup_env.py --directml")
        sys.exit(1)
    print("  Note: Installing torch-directml first — it pulls torch 2.4.1 automatically.")
    print("        Do NOT pre-install torch (causes version conflicts).")
    run([PYTHON, "-m", "pip", "install", "torch-directml"])
    _install_torch_amd_setup()
    check_env()


def install_rocm():
    banner("Installing torch for AMD ROCm + torch-amd-setup (Linux/WSL2)")
    if OS == "Windows":
        print("[ERROR] ROCm not supported on Windows natively.")
        print("        Use WSL2, or run: python setup_env.py --directml")
        sys.exit(1)
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/rocm6.1"])
    _install_torch_amd_setup()
    print()
    print("  For RX 5700 XT (gfx1010) — add to ~/.bashrc:")
    print("  export HSA_OVERRIDE_GFX_VERSION=10.3.0")
    print()
    check_env()


def install_cuda():
    banner("Installing torch for NVIDIA CUDA + torch-amd-setup")
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cu121"])
    _install_torch_amd_setup()
    check_env()


def install_mps():
    banner("Installing torch for Apple Silicon MPS + torch-amd-setup")
    if OS != "Darwin":
        print("[WARN] MPS is macOS-only. Continuing with CPU-capable torch.")
    run([PYTHON, "-m", "pip", "install", "torch"])
    _install_torch_amd_setup()
    check_env()


def install_cpu():
    banner("Installing torch (CPU only) + torch-amd-setup")
    run([PYTHON, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"])
    _install_torch_amd_setup()
    check_env()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform setup for torch-amd-setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check",    action="store_true")
    parser.add_argument("--directml", action="store_true")
    parser.add_argument("--rocm",     action="store_true")
    parser.add_argument("--cuda",     action="store_true")
    parser.add_argument("--mps",      action="store_true")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()

    if args.check:
        check_env()
    elif args.directml:
        install_directml()
    elif args.rocm:
        install_rocm()
    elif args.cuda:
        install_cuda()
    elif args.mps:
        install_mps()
    elif args.cpu:
        install_cpu()
    else:
        banner("torch-amd-setup — Auto Setup")
        best = _detect_best()
        print(f"  Detected best backend: {best}")
        if best == "directml":
            install_directml()
        elif best == "rocm":
            install_rocm()
        elif best == "cuda":
            install_cuda()
        elif best == "mps":
            install_mps()
        else:
            install_cpu()


if __name__ == "__main__":
    main()
