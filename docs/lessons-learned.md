# Lessons Learned: Building AMD ROCm + PyTorch Support from Scratch

---

## Session 2: 2026-03-23 — DirectML Benchmarking & Windows-Specific Lessons

**Context:** Real-world benchmarking of DirectML on AMD RX 5700 XT. Discovered critical Windows-only limitations and compatibility issues.

**Hardware:** AMD Radeon RX 5700 XT, Windows 11, Python 3.11.9.

### Key Discoveries

**1. Python 3.14 breaks torch-directml — must use Python 3.11**

`torch-directml` is compiled against Python 3.11 ABI and does not support 3.12+. On Windows, you must maintain a separate Python 3.11 venv for any DirectML work. Always check `py -3.11 --version` first. This is a hard ceiling, not a soft requirement.

**Workaround:** Create `.venv311` and keep it isolated. If your main project uses 3.12+, accept that DirectML is unavailable and use WSL2 + ROCm instead.

**2. torch-directml 0.2.5 requires torch 2.4.1 — install DirectML first, not torch**

Critical ordering mistake found during testing: Installing torch 2.3.0 first, then torch-directml, causes version conflicts. DirectML depends on torch 2.4.1, and pip's dependency resolver doesn't handle this gracefully.

**Correct approach:** Install torch-directml directly without pre-installing torch. Let it pull torch 2.4.1 automatically.

```bash
# WRONG (causes version hell)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml  # Pulls torch 2.4.1, conflicts with 2.3.0

# RIGHT (clean resolution)
pip install torch-directml  # Pulls torch 2.4.1 automatically
```

**3. DirectML device shows as `privateuseone:0`, not `dml:0`**

This was confusing at first because the diagnostics CLI has special-case logic to detect DirectML. When you call `str(device)`, PyTorch shows `privateuseone:0` for DirectML. This is expected and normal — it's PyTorch's way of representing custom backend devices. The GPU acceleration is working correctly.

For logging and debugging, use `torch_directml.device_name(0)` instead to get the human-readable GPU name.

**4. CTranslate2 (faster-whisper backend) does NOT support DirectML — Whisper stays on CPU**

This is a hard architectural limitation. CTranslate2, the C++ library that powers faster-whisper, only supports CUDA, ROCm, and CPU backends. It has zero DirectML support.

**Consequence:** On Windows with DirectML, Whisper inference runs entirely on CPU, completely negating GPU acceleration for speech-to-text workloads. This is a showstopper for audio pipeline projects on Windows DirectML.

**Only solution:** Use WSL2 + ROCm for GPU-accelerated Whisper on Windows AMD. There is no Windows DirectML path for Whisper.

**5. Ollama on Windows doesn't respect OLLAMA_MODELS env var if set after the parent process starts**

Tested with Ollama desktop app. Even if you set `OLLAMA_MODELS=/d/ollama_models` before launching, if the parent process (the Ollama GUI or service) started before the env var was set, Ollama will still use the default C:\Users\User\.ollama location.

**Workaround:** Use directory junctions instead of env vars. This forces Ollama to use the new location regardless of when env vars were set:

```powershell
mklink /J C:\Users\User\.ollama D:\ollama_models
```

This works because the filesystem junction is real to the OS, not dependent on environment variables.

---

## Session 1: 2026-03-23 — Original Extraction & WSL2 Setup

**Date:** 2026-03-23
**Context:** Extracting `torch-amd-setup` from a private AI audio pipeline project.
**Hardware:** AMD Radeon RX 5700 XT (gfx1010 / Navi 10), Windows 11, WSL2 Ubuntu 22.04.

This document is a raw account of every mistake made, every dependency wall hit, and every workaround discovered while getting AMD GPU acceleration working with PyTorch and Seamless M4T. Written so you don't have to spend the same time.

---

## 1. The gfx1010 problem — your GPU exists but ROCm ignores it

The single biggest source of confusion: the AMD RX 5700 XT is a capable GPU, it's supported by the AMD Adrenalin driver, and it works fine for gaming. But ROCm (AMD's GPU compute stack) has an explicit list of officially supported GPU architectures, and gfx1010 is not on it.

When you install the ROCm version of PyTorch and run `torch.cuda.is_available()`, it returns `False`. No error, no explanation — just `False`. This led to hours of assuming the ROCm install was broken, when the actual issue was a single missing environment variable:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

This has to be set **before Python imports torch**. Setting it after `import torch` does nothing. The reason: ROCm checks the GPU architecture at init time and caches the result. If the env var isn't present at that moment, the GPU is invisible for the rest of the process.

**Lesson:** If `torch.cuda.is_available()` returns False on ROCm, check the env var before anything else. Don't re-install ROCm.

---

## 2. Ubuntu 22.04 ships its own broken rocminfo

Ubuntu 22.04's default `apt` repos include `rocminfo 5.0.0-1`. This package exists to provide stub implementations of ROCm tools. When you add AMD's official ROCm 6.1 repository and try to install `rocm-hip-sdk`, apt sees the conflict and fails:

```
rocm-hip-runtime: Depends: rocminfo (= 1.0.0.60100-82~22.04)
                  but 5.0.0-1 is to be installed
```

The version numbers look backwards (5.0.0 > 1.0.0) but they're not comparable — AMD's rocminfo uses a different versioning scheme entirely. `5.0.0-1` is Ubuntu's stub; `1.0.0.60100` is AMD's real package at ROCm 6.1.

**Fix:** Remove Ubuntu's ROCm stubs before installing from AMD's repo, then pin the AMD repo to priority 1001 so it always wins in future apt operations. See [Troubleshooting](troubleshooting.md#rocm-61-install-blocked-by-ubuntus-rocminfo-500).

**Lesson:** Always purge Ubuntu's ROCm stubs before adding AMD's ROCm repo. Add the apt pin immediately.

---

## 3. set -e + grep = silent script death

When writing the automated setup script (`wsl2_rocm_setup.sh`), the script was configured with `set -euo pipefail` for safety. However, certain commands that pipe through `grep -v` would cause the entire script to silently exit with no error message.

The cause: when `apt-get -qq` runs with nothing to output (the package is already installed, or there are no packages matching), the `grep -v` that follows gets empty input and returns exit code 1 — "no lines matched the invert pattern." With `set -e`, exit code 1 from any command is fatal. The script dies silently at the first line that runs `| grep -v anything` on empty input.

The debug session was confusing because there was no error — just a prompt returning after printing one progress message.

**Fix:** `|| true` after any `grep` in a pipeline where empty output is possible. Also drop `-u` from `set -euo pipefail` if you have variables that might be unset legitimately.

**Lesson:** When a `set -e` script exits silently, check every pipe for commands that could return non-zero on "no results" — grep, awk, wc -l comparisons, etc.

---

## 4. Dependency packages silently replace your ROCm torch

Installing packages with PyPI dependencies that pin specific PyTorch versions will overwrite your ROCm build. This happened twice:

- `fairseq2==0.3.0` pins `torch==2.5.1`. pip fetched that version from PyPI, which is the standard CUDA build. ROCm build gone.
- After reinstalling ROCm torch 2.6.0, torchaudio 2.2.2 was installed separately, causing a version mismatch (`libcudart.so.13` error from the torchaudio build expecting torch 2.2.x).

Each iteration added 10–20 minutes of reinstall time and debugging.

**Lesson:** Install torch last, always. Use `--no-deps` for packages that try to pull their own torch. After any package install, verify `torch.version.hip` is still set. Consider using pip constraints or a lock file.

---

## 5. fairseq2n CPU binary doesn't exist for 0.2.1

`seamless_communication 1.0.0` requires `fairseq2==0.2.*`. `fairseq2 0.2.1` requires `fairseq2n==0.2.1` (a C extension binary). The `fairseq2n` package on PyPI ships a CUDA-linked binary — it needs `libcudart.so.12` to import.

Meta provides a CPU build server: `https://fair-src-fairseq2-build-publish.s3.amazonaws.com/whl/cpu/index.html` — but it only has builds for fairseq2n 0.3.x, not 0.2.1. So the official CPU binary for the version required by seamless_communication simply does not exist.

The solution that worked: install `nvidia-cuda-runtime-cu12` via pip, which provides `libcudart.so.12` inside the venv's site-packages, then set `LD_LIBRARY_PATH` to point at it. This lets the CUDA-linked `fairseq2n.so` load correctly even on a machine with no NVIDIA GPU.

**Lesson:** When a package claims to need CUDA but you don't have CUDA, try installing the CUDA runtime stub wheel first before assuming you need to rebuild from source.

---

## 6. torch-directml requires Python ≤3.11

`torch-directml` is Microsoft's DirectML backend for PyTorch. It provides AMD (and any DirectX 12) GPU acceleration on Windows without needing ROCm. It's genuinely useful and easy to install — but it has a hard Python version ceiling of 3.11.

This is a significant limitation because many projects now target Python 3.12+. The workaround is to maintain a separate `venv311` environment specifically for DirectML workloads. This is awkward but workable.

The underlying reason is that `torch-directml` contains compiled C extensions that were built against Python 3.11's ABI. Microsoft hasn't released 3.12 wheels as of the time of writing.

**Lesson:** Plan for a separate Python 3.11 venv on Windows if DirectML is on your path. Build your code to be venv-agnostic so switching is easy.

---

## 7. numpy 2.x breaks fairseq2 0.2.1

`fairseq2 0.2.1` was compiled against NumPy 1.x. NumPy 2.0 introduced breaking C extension ABI changes. If pip installs NumPy 2.x (which it does by default now), importing `fairseq2` crashes:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
_ARRAY_API not found
```

Fix: `pip install "numpy~=1.23" --force-reinstall`.

**Lesson:** Any package with compiled C extensions and a `numpy~=1.x` pin is going to break if pip installs numpy 2.x before it. Add an explicit numpy pin to your requirements file before installing such packages.

---

## 8. WSL2 GPU passthrough needs /dev/kfd

Even with a recent AMD driver, `/dev/kfd` (the AMD GPU compute device node) may not appear in WSL2 if:
- The AMD Adrenalin driver version is below 22.20
- The Windows version is below 10 21H2

In our case, `/dev/kfd` was missing because the driver hadn't been verified yet. This caused `rocminfo` inside WSL2 to report no agents, even though ROCm was installed correctly.

**Lesson:** Verify `/dev/kfd` exists before troubleshooting anything else. If it doesn't exist, the fix is a driver update in Windows — nothing inside WSL2 will help.

---

## 9. ROCm uses the CUDA compatibility layer — model.to("cuda") works

ROCm PyTorch exposes AMD GPUs through a CUDA compatibility layer. From the Python API perspective, `torch.cuda.is_available()` returns `True`, `torch.cuda.get_device_name(0)` returns the AMD card name, and `model.to("cuda:0")` puts the model on the AMD GPU.

This is intentional and by design. The practical consequence: code written for NVIDIA CUDA often works on AMD ROCm with zero changes. The catch is that some CUDA-specific operations (`torch.cuda.amp`, certain custom CUDA kernels) may not be supported.

**Lesson:** Don't create separate "CUDA" and "ROCm" code paths. Use `get_torch_device()` which returns `torch.device("cuda:0")` for both — the ROCm PyTorch build handles the rest.

---

## 10. The model download hits the disk hard

The SeamlessM4T v2 large model is ~8.5GB for the main checkpoint plus ~160MB for the vocoder. On first run, it downloads to `~/.cache/huggingface/hub/`. This is inside the WSL2 virtual disk, which lives on the Windows C: drive.

On a machine with limited C: drive space, this is immediately a problem. The WSL2 virtual disk is also not easily inspectable from Windows Explorer, so users may not realize a 9GB file just appeared.

**Lesson:** Warn users about the model download size before first run. Consider setting `HF_HOME` to redirect the cache to a larger drive. On a machine with an external drive, this is essential.

---

## Summary: What the setup actually requires

Getting AMD ROCm + fairseq2 + seamless_communication working requires touching at least 8 separate failure points that are not documented together anywhere:

1. Remove Ubuntu's conflicting ROCm stubs
2. Pin the ROCm apt repo to priority 1001
3. Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` before importing torch
4. Add user to `render` and `video` groups
5. Install PyTorch from the ROCm-specific index URL
6. Install `nvidia-cuda-runtime-cu12` for the CUDA stub
7. Set `LD_LIBRARY_PATH` to the stub's lib directory
8. Pin numpy to `~=1.23` before installing fairseq2

None of these steps are individually complex. But they're scattered across AMD documentation, Meta's fairseq2 GitHub issues, Ubuntu Launchpad bug reports, and Stack Overflow threads. The goal of `torch-amd-setup` is to encode as much of this as possible so future projects don't start from scratch.
