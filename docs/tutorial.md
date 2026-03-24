# Tutorial: Getting AMD GPU Acceleration with PyTorch

This tutorial walks you through every supported path for getting your AMD GPU working with PyTorch using `torch-amd-setup`. Start with the section that matches your OS.

---

## Table of Contents

1. [Before you start](#before-you-start)
2. [Windows — Path A: DirectML (Easiest)](#windows--path-a-directml-easiest)
3. [Windows — Path B: WSL2 + ROCm (Best Quality)](#windows--path-b-wsl2--rocm-best-quality)
4. [Linux — ROCm Native](#linux--rocm-native)
5. [macOS Apple Silicon — MPS](#macos-apple-silicon--mps)
6. [CPU Fallback (No GPU)](#cpu-fallback-no-gpu)
7. [Verifying your setup](#verifying-your-setup)
8. [Using torch-amd-setup in your code](#using-torch-amd-setup-in-your-code)

---

## Before you start

Check your AMD driver version. You need **Adrenalin 22.20 or newer** for WSL2 GPU passthrough. For DirectML, any recent Adrenalin release works.

**Windows (PowerShell):**
```powershell
Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion
```

**Linux:**
```bash
rocminfo | grep "Agent Type"
```

---

## Windows — Path A: DirectML (Easiest)

DirectML is Microsoft's GPU acceleration layer for machine learning. It works on any DirectX 12 GPU (AMD, NVIDIA, Intel) without needing ROCm or CUDA.

**Limitation:** `torch-directml` only supports Python 3.11 or earlier. If you're on 3.12+, use a separate venv for this.

### Step 1 — Create a Python 3.11 virtual environment

```powershell
# Check if you have Python 3.11
py -3.11 --version

# Create the venv
py -3.11 -m venv .venv311
.venv311\Scripts\activate
```

### Step 2 — Install PyTorch (CPU wheel) and DirectML

```bash
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
pip install torch-amd-setup
```

We use the CPU PyTorch wheel because `torch-directml` patches it to use the GPU — you don't need the CUDA or ROCm build.

### Step 3 — Verify

```bash
python -m torch_amd_setup
```

You should see `best_device: dml` and your GPU name in `dml_device_name`.

```python
import torch_directml
print(torch_directml.is_available())          # True
print(torch_directml.device_name(0))          # AMD Radeon RX 5700 XT
```

### Important DirectML note on float16

DirectML does not reliably support float16 operations. `torch-amd-setup` automatically returns `torch.float32` when DirectML is detected. This means your model will use more VRAM — roughly 1.5× more than float16. The RX 5700 XT has 8GB VRAM so for large models (>5B parameters) you may need to reduce batch size or use a smaller model variant.

---

## Windows — Path B: WSL2 + ROCm (Best Quality)

This path runs your Python environment inside Ubuntu on WSL2 with full ROCm GPU acceleration. It gives you float16 support, lower VRAM usage, and better performance than DirectML — but takes more initial setup.

### Requirements

- Windows 10 version 21H2 or later, or Windows 11
- AMD Adrenalin driver 22.20 or newer
- At least 20GB free disk space (Ubuntu + ROCm + model weights)

### Step 1 — Install WSL2 and Ubuntu 22.04

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot when prompted. After reboot, Ubuntu will finish installing and ask you to create a username and password. Set those and continue.

### Step 2 — Remove Ubuntu's conflicting ROCm packages

Ubuntu 22.04 ships its own `rocminfo` package (version 5.0.0) that conflicts with ROCm 6.1. Remove it first:

```bash
sudo apt-get remove -y rocminfo rocm-device-libs rocm-cmake rocm-hip-sdk rocm-hip-runtime 2>/dev/null || true
```

This step is easy to miss and causes a confusing dependency error if skipped. See [Troubleshooting](troubleshooting.md#rocm-61-install-blocked-by-ubuntu-rocminfo-500) for details.

### Step 3 — Add the ROCm 6.1 repository

```bash
# Import the ROCm GPG key
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Pin the ROCm repo to avoid Ubuntu's packages overriding it
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' | \
    sudo tee /etc/apt/preferences.d/rocm-pin

# Add the repo
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/6.1 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Install
sudo apt update && sudo apt install -y rocm-hip-sdk python3-dev python3-pip

# Add your user to GPU device groups
sudo usermod -aG render,video $USER
newgrp render
```

### Step 4 — Set the gfx1010 override (RX 5700 XT only)

The RX 5700 XT uses the gfx1010 chip architecture, which is not in ROCm's default supported GPU list. This one environment variable fixes that:

```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc
source ~/.bashrc
```

If you have a different AMD card (RX 6000/7000 series), skip this step — those are natively supported.

### Step 5 — Create a venv and install PyTorch ROCm

```bash
python3 -m venv ~/.venvs/rocm
source ~/.venvs/rocm/bin/activate

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1
pip install torch-amd-setup
```

### Step 6 — Fix the CUDA runtime stub (if needed)

If you get `ImportError: libcudart.so.12: cannot open shared object file`, install the CUDA runtime stub and set `LD_LIBRARY_PATH`:

```bash
pip install nvidia-cuda-runtime-cu12==12.1.105

CUDART_DIR=$(python3 -c "import nvidia.cuda_runtime, os; print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDART_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Make it permanent
echo "export LD_LIBRARY_PATH=\"$CUDART_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" \
    >> ~/.venvs/rocm/bin/activate
```

### Step 7 — Verify

```bash
python3 -m torch_amd_setup
```

Expected output includes `best_device: rocm` and your GPU name.

---

## Linux — ROCm Native

If you're running Ubuntu 22.04 or another supported Linux distro directly (not WSL2), the process is similar to the WSL2 path but without needing to remove conflicting packages first.

```bash
# 1. Add ROCm 6.1 repo
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/6.1 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install -y rocm-hip-sdk
sudo usermod -aG render,video $USER && newgrp render

# 2. Set override for gfx1010 GPUs (RX 5700 XT)
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
source ~/.bashrc

# 3. Install PyTorch ROCm + torch-amd-setup
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1
pip install torch-amd-setup

# 4. Verify
python3 -m torch_amd_setup
```

---

## macOS Apple Silicon — MPS

Apple's Metal Performance Shaders backend is built into standard PyTorch on Apple Silicon Macs. No extra drivers needed.

```bash
pip install torch torchvision torchaudio torch-amd-setup
python -m torch_amd_setup
# Expected: best_device: mps
```

---

## CPU Fallback (No GPU)

If you have no supported GPU, or just want to test without GPU acceleration:

```bash
pip install torch torch-amd-setup
python -m torch_amd_setup
# Expected: best_device: cpu
```

CPU mode works but is slow for large models — expect 10–30× real-time for audio/language models.

---

## Verifying your setup

Run the diagnostics CLI after any install:

```bash
python -m torch_amd_setup
```

This prints a full device report. For bug reports, always include this output.

You can also get it programmatically:

```python
from torch_amd_setup import device_info
import json
print(json.dumps(device_info(), indent=2, default=str))
```

---

## Using torch-amd-setup in your code

### Basic pattern

```python
from torch_amd_setup import get_best_device, get_torch_device, get_dtype
import torch

# Detect once at startup
device_type = get_best_device()
device      = get_torch_device(device_type)
dtype       = get_dtype(device_type)

print(f"Running on: {device_type} ({device}), dtype={dtype}")

# Apply to your model
model = YourModel()
model = model.to(device).to(dtype)

# Apply to your tensors
x = torch.randn(1, 512).to(device).to(dtype)
output = model(x)
```

### With logging

```python
import logging
logging.basicConfig(level=logging.INFO)

from torch_amd_setup import get_best_device
device = get_best_device()
# INFO     Device selected: AMD ROCm
```

### Getting install instructions in code

```python
from torch_amd_setup import get_install_guide, get_wsl2_install_guide

# Platform-appropriate instructions
print(get_install_guide())

# Full WSL2 + ROCm walkthrough
print(get_wsl2_install_guide())
```

### Overriding the detected device

```python
from torch_amd_setup import get_torch_device

# Force CPU regardless of what's available
device = get_torch_device("cpu")

# Force ROCm (will fail if not installed)
device = get_torch_device("rocm")
```
