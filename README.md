# torch-amd-setup

**Auto-detects the best PyTorch compute device for AMD GPUs** — with first-class support for cards that are not in ROCm's default allow-list (RX 5700 XT, RX 5600 XT, RX 5500 XT / gfx1010–gfx1012).

One import. No manual env var hunting. Works on Windows, Linux, WSL2, and macOS.

```python
from torch_amd_setup import get_best_device, get_torch_device, get_dtype

device_type = get_best_device()   # "rocm" | "dml" | "cuda" | "mps" | "cpu"
device      = get_torch_device()  # torch.device ready for model.to()
dtype       = get_dtype()         # torch.float16 or torch.float32
```

---

## The problem this solves

AMD GPUs that use the **gfx1010 architecture** (Navi 10 — RX 5700 XT, RX 5700, RX 5600 XT) are not in ROCm's default supported GPU list. PyTorch on ROCm will silently fall back to CPU unless you set:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

...but it has to be set *before* Python imports torch, which means you either:
- Remember to set it in every shell session, or
- Bake it into a shell script wrapper, or
- Set it in your Python script before the first `import torch`

`torch-amd-setup` handles all of that automatically. It also detects DirectML on Windows (no ROCm required), Apple MPS on macOS, NVIDIA CUDA, and falls back to CPU — so you can ship one codebase that works everywhere.

---

## Detection priority

| Priority | Backend        | Platform            | Requirement                          |
|----------|---------------|---------------------|--------------------------------------|
| 1        | NVIDIA CUDA   | Any                 | Standard `pip install torch`         |
| 2        | AMD ROCm      | Linux / WSL2        | ROCm PyTorch + AMD driver ≥22.20     |
| 3        | AMD DirectML  | Windows             | `pip install torch-directml`, Py≤3.11 |
| 4        | Apple MPS     | macOS Apple Silicon | Standard `pip install torch`         |
| 5        | CPU           | Any                 | Always available, always slow        |

---

## Install

### Quick install (one-click)

```bash
# Linux / macOS
bash install.sh            # auto-detects ROCm / CUDA / CPU
bash install.sh --rocm     # AMD GPU (Linux)
bash install.sh --cuda     # NVIDIA GPU
bash install.sh --cpu      # CPU only
bash install.sh --check    # verify environment only
```

```bat
REM Windows
install.bat                REM DirectML — requires Python 3.11
install.bat --cpu          REM CPU only — any Python version
install.bat --cuda         REM NVIDIA CUDA
install.bat --check        REM verify environment only
```

### Manual install

```bash
# 1. Install torch for your hardware (pick one):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1   # AMD ROCm (Linux)
pip install torch --index-url https://download.pytorch.org/whl/cu121     # NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cpu       # CPU only
pip install torch                                                          # macOS MPS / CUDA

# Windows DirectML (Python 3.11 only — hard ceiling):
pip install torch-directml    # pulls torch 2.4.1 automatically — do NOT pre-install torch

# 2. Install torch-amd-setup:
pip install torch-amd-setup
```

> `torch` is not a hard dependency — install the appropriate torch variant for your hardware first (see [Tutorial](docs/tutorial.md)).

---

## Quick start

```python
from torch_amd_setup import get_best_device, get_torch_device, get_dtype
import torch

device_type = get_best_device()
device      = get_torch_device(device_type)
dtype       = get_dtype(device_type)

print(f"Using: {device_type} → {device} @ {dtype}")

# Load your model
model = MyModel().to(device).to(dtype)
```

### Diagnostics CLI

```bash
python -m torch_amd_setup
```

Output:
```
── torch-amd-setup diagnostics ──────────────────────────────
  python_version            3.10.12
  platform                  Linux-6.6.x-WSL2-x86_64
  best_device               rocm
  cuda_available            True
  cuda_device_name          AMD Radeon RX 5700 XT
  cuda_vram_mb              8176
  rocm_available            True
  torch_version             2.6.0+rocm6.1
  ...
```

---

## Benchmarks

Real-world performance on **AMD Radeon RX 5700 XT** via DirectML (Windows 11, Python 3.11.9, torch 2.4.1, torch-directml 0.2.5):

| Device                          | Runtime  | TFLOPS   | Speedup |
|---------------------------------|----------|----------|---------|
| CPU (float32, AMD Ryzen 7)     | 250.4 ms | 0.55     | 1.0×    |
| AMD DirectML (float32, RX 5700 XT) | 6.2 ms   | 22.04    | **40.2× faster** |

**Key findings:**
- DirectML provides **40× speedup** over CPU for float32 workloads
- Device detection reports as `privateuseone:0` (not `dml:0`) — this is expected and normal
- Float16 support is unreliable on DirectML; float32 is the safe default

---

## Known Limitations

1. **DirectML float32-only** — No float16 support on DirectML. Models using float16 are automatically downcast to float32, which uses ~1.5× more VRAM.
2. **Python 3.11 requirement for DirectML** — `torch-directml` does not support Python 3.12 or later. Use a Python 3.11 venv if using DirectML on Windows.
3. **Whisper/CTranslate2 incompatibility** — CTranslate2 (the backend for faster-whisper) does not support DirectML. Whisper inference must run on CPU even with DirectML available. For GPU-accelerated Whisper on AMD, use ROCm on Linux/WSL2.
4. **GPU memory overhead** — DirectML uses roughly 1.5× more VRAM than ROCm for the same model due to float32-only execution and driver overhead.

---

## API Reference

### `get_best_device() → str`
Returns the best available device type as a string: `"cuda"`, `"rocm"`, `"dml"`, `"mps"`, or `"cpu"`.

### `get_torch_device(device_type=None) → torch.device`
Returns a `torch.device` object (or a DirectML device object for `"dml"`) ready for `model.to()`. If `device_type` is `None`, calls `get_best_device()` automatically.

### `get_dtype(device_type=None) → torch.dtype`
Returns `torch.float16` for CUDA/ROCm/MPS, and `torch.float32` for DirectML/CPU. DirectML float16 support is unreliable; this keeps you safe.

### `device_info() → dict`
Returns a diagnostic dictionary with all detected hardware info. Useful for logging and bug reports.

### `get_install_guide() → str`
Returns platform-appropriate install instructions as a formatted string.

### `get_wsl2_install_guide() → str`
Returns the full WSL2 + ROCm setup walkthrough for AMD GPUs on Windows.

### `AMD_ROCM_ENV: dict`
The environment variable overrides applied for gfx1010 support. You can inspect or override these before calling `get_best_device()`.

---

## AMD GPU compatibility

| GPU                     | Architecture | HSA Override   | Tested |
|-------------------------|-------------|----------------|--------|
| RX 5700 XT              | gfx1010     | `10.3.0`       | ✅     |
| RX 5700                 | gfx1010     | `10.3.0`       | ✅     |
| RX 5600 XT              | gfx1010     | `10.3.0`       | ✅     |
| RX 5500 XT              | gfx1011     | `10.3.0`       | ⚠️ reported |
| RX 6000 series (gfx1030+) | RDNA2    | Not needed     | ✅ native ROCm |
| RX 7000 series (gfx1100+) | RDNA3    | Not needed     | ✅ native ROCm |

If your card isn't listed, check `GFX_OVERRIDE_MAP` in `detect.py` and open a PR.

---

## Windows users: DirectML vs WSL2

| Feature              | DirectML           | WSL2 + ROCm        |
|----------------------|--------------------|--------------------|
| Setup difficulty     | Easy               | Medium             |
| float16 support      | ❌ (float32 only) | ✅                 |
| Python version limit | 3.11 max           | Any                |
| GPU memory usage     | ~1.5× higher       | Native             |
| Best for             | Quick experiments  | Production workloads |

---

## Troubleshooting

### Windows

**torch-directml import fails / wrong torch version**
Install directml *without* pre-installing torch. It pulls the correct torch 2.4.1 automatically:
```bat
pip uninstall torch -y
pip install torch-directml
```

**Python 3.12+ — torch-directml not available**
DirectML requires Python ≤ 3.11. Create a 3.11 venv:
```bat
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install torch-directml torch-amd-setup
```

**`get_best_device()` returns "cpu" on Windows with AMD GPU**
DirectML was not detected. Check: `python -m torch_amd_setup` — if DML is missing, install it:
```bat
pip install torch-directml
```

**`privateuseone:0` device string**
Normal and expected. This is how PyTorch represents DirectML custom backends.

**Whisper stays on CPU even with GPU available**
CTranslate2 (faster-whisper backend) has no DirectML support — this is a hard architectural limit. For GPU-accelerated Whisper on AMD, use WSL2 + ROCm.

---

### Linux (ROCm)

**`torch.cuda.is_available()` returns False**
You likely installed the CPU torch wheel. Check:
```bash
python3 -c "import torch; print(torch.__version__)"
# Should show: 2.x.x+rocm6.1 (not +cpu)
```
If it shows `+cpu`, reinstall:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
```

**RX 5700 XT not detected / falls back to CPU**
The gfx1010 override is missing. Add to `~/.bashrc`:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
Or set it in your Python script *before* importing torch:
```python
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import torch  # now picks up the override
```
`torch-amd-setup` does this automatically — just call `get_best_device()` first.

**`rocminfo` shows version 5.0.0 and fails**
Ubuntu ships a stub `rocminfo`. Remove it before installing ROCm:
```bash
sudo apt remove rocminfo
```

**`/dev/kfd: Permission denied`**
```bash
sudo usermod -aG render,video $USER
# Log out and back in
```

**`set -e` + script exits silently**
If using `grep` inside a `set -e` script and grep finds no match (exit code 1), the script dies silently. Use `grep ... || true` to avoid this.

**numpy 2.x ABI break**
torch ≤ 2.4 requires numpy <2.0:
```bash
pip install "numpy<2.0"
```

---

### macOS

**MPS not available**
Requires macOS 12.3+ with Apple Silicon (M1/M2/M3). Intel Macs do not have MPS — use CPU.
```bash
python3 -c "import platform; print(platform.mac_ver())"
```

---

### All Platforms

**Verify your setup at any time:**
```bash
python -m torch_amd_setup        # full diagnostics
bash install.sh --check          # Linux/macOS
install.bat --check              # Windows
```

---

## Contributing

PRs welcome. Especially interested in:
- Verified gfx override values for additional GPU models
- ROCm 6.2+ compatibility reports
- Windows DirectML on NVIDIA/Intel test results

Please open an issue before large PRs.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Background

This package was extracted from a private AI music pipeline project. The gfx1010 ROCm workaround was discovered the hard way — through several hours of cascading PyTorch installs, ROCm SDK conflicts, and dependency hell. The goal is that nobody else has to spend that time.

See [docs/lessons-learned.md](docs/lessons-learned.md) for the full story.
