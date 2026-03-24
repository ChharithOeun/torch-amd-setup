"""
Basic usage example for torch-amd-setup.

Run this after installing:
    pip install torch-amd-setup
    pip install torch  # or your hardware-specific variant
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

from torch_amd_setup import (
    get_best_device,
    get_torch_device,
    get_dtype,
    device_info,
)

# ── 1. Detect best device ─────────────────────────────────────────────────────
device_type = get_best_device()
print(f"\nBest device: {device_type}")

# ── 2. Get torch.device object ────────────────────────────────────────────────
device = get_torch_device(device_type)
print(f"torch.device: {device}")

# ── 3. Get recommended dtype ──────────────────────────────────────────────────
dtype = get_dtype(device_type)
print(f"dtype:        {dtype}")

# ── 4. Use with a model ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    model = nn.Linear(512, 512)
    model = model.to(device).to(dtype)

    x = torch.randn(1, 512, dtype=dtype).to(device)
    output = model(x)
    print(f"\nForward pass OK — output shape: {output.shape}")
except Exception as e:
    print(f"\nForward pass skipped: {e}")

# ── 5. Full diagnostics ───────────────────────────────────────────────────────
print("\n── Full device info ──────────────────────────────────────────")
import json
info = device_info()
for k, v in info.items():
    print(f"  {k:<25} {v}")
