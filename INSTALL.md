# Installation Guide

Complete step-by-step instructions for setting up PyTorch with DirectML on Windows.

## Prerequisites

- Windows 10 or later
- Python 3.10 or higher
- AMD, Intel, or Nvidia GPU (with DirectML support)
- Updated GPU drivers

## Step 1: Install Python

Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

**Important:** During installation, check "Add Python to PATH"

Verify installation:
```bash
python --version
```

## Step 2: Update GPU Drivers

### AMD GPUs
- Download the latest AMD GPU driver from [AMD Support](https://www.amd.com/en/support)
- Install and restart your computer

### Intel GPUs
- Download the latest Intel Arc GPU driver from [Intel Download Center](https://www.intel.com/content/www/us/en/download-center/home.html)
- Install and restart your computer

### Nvidia GPUs
- DirectML works with Nvidia GPUs, but CUDA is recommended for better performance
- If using DirectML on Nvidia: [Nvidia Driver Downloads](https://www.nvidia.com/Download/driverDetails.aspx)

## Step 3: Clone the Repository

```bash
git clone https://github.com/ChharithOeun/torch-amd-setup.git
cd torch-amd-setup
```

## Step 4: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

On PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch-directml` - DirectML backend for PyTorch
- `diffusers` - Stable Diffusion models
- `transformers` - HuggingFace models
- `accelerate` - Distributed training utilities

## Step 6: Verify Installation

```bash
python scripts/verify_gpu.py
```

Expected output:
```
============================================================
torch-amd-setup GPU Verification
============================================================

✓ PyTorch installed: 2.0.1
✓ torch-directml available
✓ DirectML device count: 1
✓ DirectML device: dml:0
✓ Matrix multiplication (512x512): 12.34ms
✓ Neural network forward pass: 5.67ms
✓ Estimated available VRAM: ~8192MB+

============================================================
Summary
============================================================
Passed: 7/7

✓ GPU setup verified! You can now run training scripts.
```

## Step 7: Test with Hello GPU Demo

```bash
python scripts/hello_gpu.py
```

Expected output shows tensor operations and training loop completing successfully.

## Common Pitfalls and Fixes

### Issue: ModuleNotFoundError for torch or torch_directml

**Cause:** Dependencies not installed

**Fix:**
```bash
pip install -r requirements.txt
```

### Issue: CUDA version of PyTorch conflicts with DirectML

**Error:** "RuntimeError: No available compute devices"

**Cause:** You have PyTorch with CUDA (e.g., `torch-cu118`) installed alongside torch-directml

**Fix:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch-directml
```

### Issue: Float16 operations fail or fall back to CPU

**Cause:** Not all ops support float16 on DirectML

**Solution:** Use float32 for training:
```python
import torch
import torch_directml

device = torch_directml.device()

# Use float32 (default)
x = torch.randn(10, 10, dtype=torch.float32).to(device)

# Avoid float16 unless necessary
# x = torch.randn(10, 10, dtype=torch.float16).to(device)  # May fall back to CPU
```

### Issue: "No suitable graphics device found"

**Cause:** GPU drivers not installed or outdated

**Fix:**
1. Update GPU drivers (AMD, Intel, or Nvidia)
2. Restart computer
3. Run `python scripts/verify_gpu.py` again

### Issue: Out of Memory (OOM)

**Solution:**
1. Reduce batch size
2. Use smaller models
3. Clear GPU memory:
   ```python
   import torch
   torch.cuda.empty_cache()  # CUDA-style cleanup (auto on DirectML)
   ```

### Issue: Slow performance

**Cause:** Operations falling back to CPU

**Solution:**
1. Check with `python scripts/benchmark.py`
2. Verify GPU is being used in `verify_gpu.py`
3. Use explicit device placement: `.to(device)`
4. Avoid unsupported dtypes (float16)

## Verification Checklist

- [ ] Python 3.10+ installed
- [ ] GPU drivers updated
- [ ] torch-directml installed without CUDA PyTorch
- [ ] `verify_gpu.py` passes all checks
- [ ] `hello_gpu.py` completes training loop
- [ ] Benchmark shows speedup vs CPU

## Next Steps

- Run training scripts: `python scripts/hello_gpu.py`
- Benchmark performance: `python scripts/benchmark.py`
- Use in your own projects: `import torch_directml; device = torch_directml.device()`

## Resources

- [torch-directml Official Docs](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-release-notes)
- [AMD Windows Toolkit](https://github.com/ChharithOeun/amd-windows-toolkit)
- [PyTorch Docs](https://pytorch.org/docs/)

## Support

Found an issue? Report it on [GitHub Issues](https://github.com/ChharithOeun/torch-amd-setup/issues)

## Buy Me A Coffee

If this helped you set up PyTorch on AMD GPU, consider supporting my work:

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20Me%20A%20Coffee-FFDD00?style=flat&logoColor=white)](https://buymeacoffee.com/chharith)
