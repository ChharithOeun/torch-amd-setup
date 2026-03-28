#!/usr/bin/env bash
# install.sh — torch-amd-setup cross-platform installer
# Works on: Linux (ROCm/CUDA/CPU) and macOS (MPS/CPU)
# For Windows: use install.bat instead
#
# Usage:
#   bash install.sh            # auto-detect best backend
#   bash install.sh --rocm     # AMD GPU via ROCm (Linux)
#   bash install.sh --cuda     # NVIDIA GPU via CUDA
#   bash install.sh --cpu      # CPU only
#   bash install.sh --check    # verify environment only

set -e

ROCM=0; CUDA=0; CPU=0; CHECK=0

for arg in "$@"; do
    case "$arg" in
        --rocm)  ROCM=1  ;;
        --cuda)  CUDA=1  ;;
        --cpu)   CPU=1   ;;
        --check) CHECK=1 ;;
    esac
done

OS="$(uname -s)"
PYTHON="${PYTHON:-python3}"

echo "════════════════════════════════════════════════════"
echo "  torch-amd-setup installer"
echo "  OS: $OS  |  Python: $($PYTHON --version 2>&1)"
echo "════════════════════════════════════════════════════"

if [ "$CHECK" -eq 1 ]; then
    echo ""
    echo "Checking environment..."
    $PYTHON -c "
import sys, platform
print('Python :', sys.version)
print('OS     :', platform.system(), platform.version())
try:
    import torch
    print('torch  :', torch.__version__)
    print('CUDA   :', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU    :', torch.cuda.get_device_name(0))
    print('HIP    :', getattr(torch.version, 'hip', None))
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print('MPS    :', mps)
except ImportError:
    print('torch  : NOT INSTALLED')
try:
    from torch_amd_setup import get_best_device, device_info
    print('amd-setup:', get_best_device())
except ImportError:
    print('torch-amd-setup: NOT INSTALLED')
"
    exit 0
fi

# ── Auto-detect ───────────────────────────────────────────────────────────
if [ "$ROCM" -eq 0 ] && [ "$CUDA" -eq 0 ] && [ "$CPU" -eq 0 ]; then
    if command -v rocminfo &>/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q gfx; then
        echo "[AUTO] ROCm detected → installing torch+ROCm"
        ROCM=1
    elif command -v nvidia-smi &>/dev/null 2>&1; then
        echo "[AUTO] NVIDIA GPU detected → installing torch+CUDA"
        CUDA=1
    else
        echo "[AUTO] No GPU found → installing CPU-only torch"
        CPU=1
    fi
fi

# ── Step 1: Install torch backend ────────────────────────────────────────
if [ "$ROCM" -eq 1 ]; then
    echo "[1/3] Installing torch for ROCm 6.1..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

elif [ "$CUDA" -eq 1 ]; then
    echo "[1/3] Installing torch for CUDA 12.1..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121

elif [ "$CPU" -eq 1 ]; then
    echo "[1/3] Installing torch (CPU only)..."
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# ── Step 2: Install torch-amd-setup ──────────────────────────────────────
echo "[2/3] Installing torch-amd-setup..."
$PYTHON -m pip install torch-amd-setup

# ── Step 3: Verify ───────────────────────────────────────────────────────
echo "[3/3] Verifying install..."
bash "$0" --check

# ── AMD env var hints ─────────────────────────────────────────────────────
if [ "$ROCM" -eq 1 ]; then
    echo ""
    echo "  AMD GPU setup notes:"
    echo "  - RX 5700 XT (gfx1010): add to ~/.bashrc:"
    echo "      export HSA_OVERRIDE_GFX_VERSION=10.3.0"
    echo "  - Check your GPU: rocminfo | grep 'Marketing Name'"
    echo "  - Verify torch sees GPU: python3 -c 'import torch; print(torch.cuda.is_available())'"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  Done! Quick test:"
echo "  python3 examples/basic_usage.py"
echo "════════════════════════════════════════════════════"
