# Troubleshooting Guide

Every error in this document was encountered during real development of this library. If you hit something not covered here, open an issue and include the output of `python -m torch_amd_setup`.

---

## Table of Contents

1. [ROCm 6.1 install blocked by Ubuntu's rocminfo 5.0.0](#rocm-61-install-blocked-by-ubuntus-rocminfo-500)
2. [Script exits silently at Step 1 (grep + set -e)](#script-exits-silently-at-step-1-grep--set--e)
3. [ImportError: libcudart.so.12 not found](#importerror-libcudartso12-not-found)
4. [fairseq2 / other packages downgrade torch to non-ROCm build](#fairseq2--other-packages-downgrade-torch-to-non-rocm-build)
5. [numpy ABI incompatibility (_ARRAY_API not found)](#numpy-abi-incompatibility-_array_api-not-found)
6. [/dev/kfd missing — GPU not visible to WSL2](#devkfd-missing--gpu-not-visible-to-wsl2)
7. [torch.cuda.is_available() returns False on ROCm build](#torchcudais_available-returns-false-on-rocm-build)
8. [DirectML not available / torch_directml ImportError](#directml-not-available--torch_directml-importerror)
9. [torch-directml not detected / falling back to CPU](#torch-directml-not-detected--falling-back-to-cpu)
10. [Device shows as privateuseone:0 instead of dml:0](#device-shows-as-privateuseone0-instead-of-dml0)
11. [diffusers pipeline silently runs on CPU with DirectML (device string vs object)](#diffusers-pipeline-silently-runs-on-cpu-with-directml-device-string-vs-object)
12. [Whisper always runs on CPU even with AMD GPU](#whisper-always-runs-on-cpu-even-with-amd-gpu)
13. [Wrong WSL distro — Python not found](#wrong-wsl-distro--python-not-found)
14. [git index.lock error on NTFS mount](#git-indexlock-error-on-ntfs-mount)

---

## ROCm 6.1 install blocked by Ubuntu's rocminfo 5.0.0

**Error:**
```
rocm-hip-runtime: Depends: rocminfo (= 1.0.0.60100-82~22.04)
                  but 5.0.0-1 is to be installed
```

**Cause:** Ubuntu 22.04's default package repos ship `rocminfo 5.0.0-1`. When you add the ROCm 6.1 repo, apt sees both versions and picks the wrong one, causing a dependency conflict.

**Fix:**
```bash
# Remove Ubuntu's ROCm stubs first
sudo apt-get remove -y rocminfo rocm-device-libs rocm-cmake rocm-hip-sdk rocm-hip-runtime 2>/dev/null || true

# Pin the ROCm 6.1 repo to priority 1001 so it always wins
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' | \
    sudo tee /etc/apt/preferences.d/rocm-pin

# Now install
sudo apt update && sudo apt install -y rocm-hip-sdk
```

---

## Script exits silently at Step 1 (grep + set -e)

**Symptom:** A shell script with `set -euo pipefail` prints one line then exits with no error message.

**Cause:** When `apt-get -qq` produces no output (nothing to filter), `grep -v "^some-pattern:"` returns exit code 1 (no matches found). With `set -e`, exit code 1 from any command kills the script immediately — even a grep with nothing to filter is treated as a fatal error.

**Fix:** Change `set -euo pipefail` to `set -eo pipefail` (drop `-u` if you have unset vars) and add `|| true` after any grep pipes that might produce no output:

```bash
# Before (breaks)
apt-get -qq install something | grep -v "^debconf:"

# After (safe)
apt-get -qq install something | grep -v "^debconf:" || true
```

---

## ImportError: libcudart.so.12 not found

**Error:**
```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

**Cause:** Some packages ship C extensions that are compiled against CUDA 12's runtime library (`libcudart.so.12`). In a CPU-only or ROCm environment, this library doesn't exist at the system level.

**Fix:** Install the NVIDIA CUDA 12 runtime stub via pip and point `LD_LIBRARY_PATH` at it. This gives the dynamic linker the `.so` file it needs without requiring an actual NVIDIA GPU:

```bash
pip install nvidia-cuda-runtime-cu12==12.1.105

CUDART_DIR=$(python3 -c "
import nvidia.cuda_runtime, os
print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib'))
")

export LD_LIBRARY_PATH="$CUDART_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Make permanent in your venv's activate script
echo "export LD_LIBRARY_PATH=\"$CUDART_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" \
    >> ~/.venvs/your-venv/bin/activate
```

Why the pip wheel works: `nvidia-cuda-runtime-cu12` ships the actual `libcudart.so.12` inside its package directory. The standard system library path doesn't include venv site-packages, so you need to set `LD_LIBRARY_PATH` explicitly.

---

## fairseq2 / other packages downgrade torch to non-ROCm build

**Symptom:** You install `torch+rocm6.1` successfully, then install another package, and suddenly `torch.version.hip` is `None` and GPU is gone.

**Cause:** Many packages pin specific torch versions (e.g. `fairseq2==0.3.0` pins `torch==2.5.1`). When pip resolves dependencies, it fetches that pinned version from PyPI — which is the standard CUDA/CPU build, not your ROCm build. It silently overwrites your ROCm torch.

**Fix:** Install torch last, with `--no-deps` for the problematic package, then reinstall torch ROCm after:

```bash
# Install the package without letting it touch torch
pip install some-package --no-deps

# Reinstall the ROCm build on top
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 \
    --force-reinstall

# Verify ROCm is back
python3 -c "import torch; print(torch.version.hip)"
```

Alternatively, use constraints:
```bash
pip install some-package --constraint <(echo "torch==2.6.0+rocm6.1")
```

---

## numpy ABI incompatibility (_ARRAY_API not found)

**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
_ARRAY_API not found
```

**Cause:** Some packages (including `fairseq2==0.2.*`) were compiled against NumPy 1.x. NumPy 2.x broke the C extension ABI, so the compiled `.so` crashes when it tries to use the old API.

**Fix:** Pin numpy to a 1.x version:
```bash
pip install "numpy~=1.23" --force-reinstall
```

If another package in your environment requires numpy 2.x, you have a genuine conflict and need separate venvs for each package.

---

## /dev/kfd missing — GPU not visible to WSL2

**Symptom:** `rocminfo` in WSL2 reports no agents, or `torch.cuda.is_available()` returns False even with ROCm PyTorch installed.

**Cause:** `/dev/kfd` is the AMD GPU compute device node. It only appears in WSL2 if:
1. Your AMD Adrenalin driver on Windows is version 22.20 or newer
2. Your Windows version supports WSL2 GPU passthrough (Win10 21H2+ or Win11)

**Diagnosis:**
```bash
ls -la /dev/kfd /dev/dri/renderD*
```

If `/dev/kfd` doesn't exist, the driver isn't passing the GPU through.

**Fix:**
1. Check your driver version in AMD Software (Adrenalin): open it and look at **Driver & Software → Current Version**
2. If below 22.20, click **Manage Updates** and update
3. After updating, restart Windows and re-open your WSL2 terminal
4. Verify: `ls /dev/kfd` should now exist

---

## torch.cuda.is_available() returns False on ROCm build

**Symptom:** You installed `torch+rocm6.1` but `torch.cuda.is_available()` still returns `False`.

**Possible causes and fixes:**

**1. HSA_OVERRIDE_GFX_VERSION not set (gfx1010 cards)**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 -c "import torch; print(torch.cuda.is_available())"
```

**2. /dev/kfd not accessible**
```bash
sudo usermod -aG render,video $USER
newgrp render
# Log out and back in, then retry
```

**3. Wrong torch build installed**
```bash
# Verify your torch has ROCm
python3 -c "import torch; print(torch.__version__, torch.version.hip)"
# Should print: 2.x.x+rocm6.1   6.1.something
# If it prints: 2.x.x+cu121  None — you have CUDA torch, not ROCm
```

If you have the wrong build:
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 \
    --force-reinstall
```

---

## DirectML not available / torch_directml ImportError

**Symptom:** `get_best_device()` returns `"cpu"` on Windows even with an AMD GPU.

**Check 1 — Python version:**
```python
import sys; print(sys.version_info[:2])
# (3, 12) or higher → torch-directml is not supported on Python 3.12+
# Solution: use a Python 3.11 venv
```

**Check 2 — Is torch-directml installed?**
```bash
pip show torch-directml
# Not found → pip install torch-directml
```

**Check 3 — Is torch-directml getting a compatible torch?**
```python
import torch_directml
print(torch_directml.is_available())
```

If this raises an error about incompatible torch versions, reinstall torch first:
```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
```

---

## torch-directml not detected / falling back to CPU

**Symptom:** `get_best_device()` returns `"cpu"` on Windows even with an AMD GPU, or `python -m torch_amd_setup` shows `best_device: cpu`.

**Check 1 — Python version (CRITICAL):**
```bash
python --version
# If output shows 3.12 or higher: torch-directml is not supported on Python 3.12+
# Solution: Create a separate Python 3.11 venv
py -3.11 -m venv .venv311
.venv311\Scripts\activate
```

**Check 2 — Is torch-directml installed?**
```bash
pip show torch-directml
# Not found → pip install torch-directml
```

**Check 3 — Correct installation order (important)**
Install `torch-directml` directly without installing torch first. DirectML will pull the correct torch version automatically:
```bash
# WRONG: installs incompatible torch versions
pip install torch==2.3.0
pip install torch-directml

# CORRECT: let DirectML pull torch 2.4.1
pip install torch-directml
```

**Check 4 — Verify DirectML is working:**
```python
import torch_directml
print(torch_directml.is_available())      # Should be True
print(torch_directml.device_name(0))      # Should show your GPU name
```

---

## Device shows as privateuseone:0 instead of dml:0

**Symptom:** `str(device)` or printing `get_torch_device()` shows `privateuseone:0` instead of the expected `dml:0`.

**This is normal for DirectML.** The device string `privateuseone:0` indicates DirectML is working correctly — it's PyTorch's internal representation for custom GPU backends. The GPU acceleration is active regardless of the device string name.

**How to use it:**
```python
from torch_amd_setup import get_torch_device
import torch

device = get_torch_device()
print(str(device))  # Prints: privateuseone:0

# This works correctly — your model will use DirectML acceleration
model = MyModel().to(device)
```

If you need the GPU name for logging, use this instead:
```python
import torch_directml
gpu_name = torch_directml.device_name(0)  # Returns: "AMD Radeon RX 5700 XT"
```

---

## diffusers pipeline silently runs on CPU with DirectML (device string vs object)

**Symptom:** DirectML is working (benchmark confirms GPU acceleration, `torch_directml.is_available()` returns `True`), but a diffusers pipeline (`StableDiffusionXLPipeline`, `StableDiffusionPipeline`, etc.) generates images on CPU — taking 5–15 minutes per image instead of seconds. The device badge or log shows `cpu` even though DirectML detection succeeded.

**Root cause:** Passing the DirectML device as a **string** to `pipe.to()` silently fails. The diffusers pipeline iterates all sub-components (UNet, VAE, text encoders) and calls `.to(device)` on each. When `device` is the string `"privateuseone:0"`, PyTorch cannot parse it as a valid device, raises an internal exception, and the pipeline falls back to CPU. Because the exception is caught internally and no warning is printed, the failure is completely invisible.

The critical distinction:
```python
# WRONG — string form, fails silently inside diffusers
device_str = str(torch_directml.device())  # "privateuseone:0"
pipe = pipe.to(device_str)                 # silently runs on CPU

# CORRECT — device object, works
device_obj = torch_directml.device()      # keep the actual object
pipe = pipe.to(device_obj)                # runs on GPU
```

**Why it's hard to catch:** Benchmarks using simple tensor ops work fine with the string form. `torch.randn(n, n).to("privateuseone:0")` succeeds. Only a full pipeline `.to()` that iterates all sub-components exposes this failure. If you proved your GPU works with a matmul benchmark but your diffusion pipeline is still on CPU, this is the reason.

**Complete working pattern for DirectML + diffusers:**

```python
import torch
import torch_directml
from diffusers import StableDiffusionXLPipeline

# Step 1: get device OBJECT (not string)
dml_device = torch_directml.device()
print(f"DirectML device: {dml_device}")       # privateuseone:0
print(f"GPU name: {torch_directml.device_name(0)}")  # AMD Radeon RX 5700 XT

# Step 2: load with float32 — DirectML does not support float16
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,   # NOT float16
    use_safetensors=True,
    # Do NOT pass variant="fp16" — that variant uses fp16 weights
)
pipe.enable_attention_slicing()  # reduces VRAM usage

# Step 3: move pipeline using the OBJECT
pipe = pipe.to(dml_device)

# Step 4: verify GPU is active
print(pipe.unet.device)   # should print: privateuseone:0

# Step 5: use cpu Generator — DirectML doesn't support Generator on device
generator = torch.Generator(device="cpu")
generator.manual_seed(42)

result = pipe(
    prompt="a photo of an astronaut riding a horse",
    generator=generator,
    num_inference_steps=25,
)
result.images[0].save("output.png")
```

**In torch-amd-setup:** `get_torch_device()` returns the device object directly, so it's always safe to use with `.to()`. The issue only appears when you call `str()` on the returned device and then pass that string to `.to()`:

```python
from torch_amd_setup import get_torch_device

device = get_torch_device()     # SAFE — returns device object
pipe = pipe.to(device)          # works correctly

# Don't do this:
pipe = pipe.to(str(device))     # FAILS silently — passes "privateuseone:0" string
```

**Expected performance after fix:** On AMD RX 5700 XT, SDXL at 25 steps drops from ~10 minutes (CPU) to ~2–4 minutes (DirectML, float32). For float16 speeds, use WSL2 + ROCm.

---

## Whisper always runs on CPU even with AMD GPU

**Symptom:** Using faster-whisper or OpenAI Whisper on Windows with AMD GPU, but GPU is never used — inference stays on CPU.

**Root cause:** CTranslate2 (the backend used by faster-whisper) does not support DirectML. Even if you have an AMD GPU and DirectML installed, Whisper must run on CPU.

**This is a known limitation** with no workaround on Windows DirectML.

**Workaround — Use WSL2 + ROCm instead:**
If you need GPU-accelerated Whisper on Windows with AMD, use WSL2 + ROCm:
1. Set up WSL2 with Ubuntu 22.04 and ROCm 6.1 (see [Tutorial](tutorial.md#windows--path-b-wsl2--rocm-best-quality))
2. Install faster-whisper in WSL2 — it will automatically use ROCm/GPU
3. Call the WSL2 venv from Windows if needed via a wrapper script

**Workaround — Reduce model size for CPU:**
If CPU inference is acceptable, use the `base` or `tiny` Whisper model instead of `large` to reduce inference time.

---

## Wrong WSL distro — Python not found

**Symptom:** After running `wsl`, you get a minimal shell with no Python, or commands behave unexpectedly.

**Cause:** Typing `wsl` opens your *default* WSL distribution, which may not be Ubuntu 22.04 (especially if you have multiple distros installed, or if the default is a minimal system distro).

**Fix:**
```powershell
# List all installed distros
wsl --list --verbose

# Open specifically Ubuntu 22.04
wsl -d Ubuntu-22.04

# Set Ubuntu 22.04 as default so bare `wsl` opens it
wsl --set-default Ubuntu-22.04
```

---

## git index.lock error on NTFS mount

**Symptom:** When running git commands on a repository located on `/mnt/c/` (the Windows C: drive mounted inside WSL2), you get:
```
fatal: Unable to create '.git/index.lock': File exists.
```

And `rm .git/index.lock` fails with permission denied.

**Cause:** NTFS file locking semantics differ from Linux. WSL2's NTFS mount can leave lock files that Linux tools can't remove.

**Workaround:**
```bash
# Copy the git index to a Linux filesystem, commit from there
cp .git/index /tmp/git-index-backup

GIT_INDEX_FILE=/tmp/git-index-backup git add your-file.py
TREE=$(GIT_INDEX_FILE=/tmp/git-index-backup git write-tree)
COMMIT=$(echo "your commit message" | git commit-tree $TREE -p HEAD)
python3 -c "open('.git/refs/heads/master','w').write('$COMMIT\n')"
```

**Better long-term fix:** Clone your repo to a native Linux path inside WSL2 (`~/projects/` rather than `/mnt/c/`) and sync files there. NTFS mount performance and locking issues are a known WSL2 limitation.
