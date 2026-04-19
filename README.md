# torch-amd-setup

[![CI](https://github.com/ChharithOeun/torch-amd-setup/actions/workflows/ci.yml/badge.svg)](https://github.com/ChharithOeun/torch-amd-setup/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20Me%20A%20Coffee-FFDD00?style=flat&logoColor=white)](https://buymeacoffee.com/chharith)

**PyTorch on AMD GPU Windows via DirectML** — Training, inference, and research workloads on AMD/Intel GPUs.

## Description

`torch-amd-setup` provides a complete environment setup guide and automation scripts for running PyTorch workloads on AMD GPUs using Microsoft's **torch-directml** backend. Eliminates the complexity of GPU driver setup and PyTorch configuration on Windows.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ChharithOeun/torch-amd-setup.git
cd torch-amd-setup

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU setup
python scripts/verify_gpu.py

# 4. Run a simple demo
python scripts/hello_gpu.py
```

## What is torch-directml?

**torch-directml** is Microsoft's GPU acceleration backend for PyTorch. It enables PyTorch to run on AMD, Intel, and Nvidia GPUs on Windows via DirectX 12 and DirectML. Unlike CUDA (Nvidia-only), DirectML is cross-vendor and works natively on Windows without special driver installations.

## Features

- ✅ Full tensor operations on GPU
- ✅ Model training and inference
- ✅ NumPy interoperability
- ✅ Stable Diffusion pipeline support
- ✅ ONNX export capability
- ✅ Cross-platform (Windows, Linux coming)

## Usage

### Basic Tensor Operations

```python
import torch
import torch_directml

# Get DirectML device
dml = torch_directml.device()

# Create and move tensors to GPU
x = torch.randn(3, 3).to(dml)
y = torch.randn(3, 3).to(dml)

# Perform operations on GPU
z = torch.matmul(x, y)
print(z)
```

### Moving Tensors to DirectML Device

```python
import torch
import torch_directml

device = torch_directml.device()

# Explicit device movement
tensor = torch.randn(100, 100).to(device)

# Operations stay on GPU
result = tensor @ tensor.T
```

### Training a Simple Model

```python
import torch
import torch.nn as nn
import torch_directml

device = torch_directml.device()

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

for epoch in range(100):
    x = torch.randn(32, 10).to(device)
    y = torch.randn(32, 1).to(device)
    
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
```

## Limitations

- Some operations don't support autograd on all dtypes (float16 fallback to float32)
- Certain operations may fall back to CPU automatically
- No NCCL support for distributed multi-GPU training yet
- Best performance with float32 tensors

## Known Working Use Cases

- ✅ Stable Diffusion pipelines (inference)
- ✅ ONNX model export
- ✅ Transformer inference (HuggingFace)
- ✅ Basic neural network training
- ✅ Computer vision models (ResNet, etc.)

## VRAM Tips

- Monitor VRAM usage with `scripts/verify_gpu.py`
- Start with smaller batch sizes and increase gradually
- Use mixed precision (float32 primary, float16 carefully)
- Allocate tensors explicitly to device to avoid CPU fallback
- Use `torch.cuda.empty_cache()` equivalent: `torch_directml` handles cleanup automatically

## Related

- [amd-windows-toolkit](https://github.com/ChharithOeun/amd-windows-toolkit) — AMD GPU driver setup and utilities
- [Official torch-directml Docs](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-release-notes)

## License

MIT License © 2024 Chharith Oeun

## Support

Love this project? Consider buying me a coffee!

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20Me%20A%20Coffee-FFDD00?style=flat&logoColor=white)](https://buymeacoffee.com/chharith)
