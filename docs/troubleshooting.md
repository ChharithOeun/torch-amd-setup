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
9. [Wrong WSL distro — Python not found](#wrong-wsl-distro--python-not-found)
10. [git index.lock error on NTFS mount](#git-indexlock-error-on-ntfs-mount)

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
