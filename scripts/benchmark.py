"""
Benchmark torch-directml performance vs CPU.

Usage:
    python scripts/benchmark.py

Runs matrix multiplication, element-wise operations, and MLP forward passes
on both DirectML and CPU, printing speedup ratios.
"""

import time
import sys

def benchmark_matmul(device, sizes, name):
    """Benchmark matrix multiplication at various sizes."""
    import torch

    print(f"\nMatrix Multiplication Benchmark ({name})")
    print("-" * 50)

    for size in sizes:
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)

        # Warmup
        for _ in range(3):
            _ = torch.matmul(a, b)

        # Time
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        elapsed = time.time() - start

        avg_ms = (elapsed / 10) * 1000
        print(f"  {size}x{size}: {avg_ms:.2f}ms")

    return avg_ms

def benchmark_elementwise(device, size, name):
    """Benchmark element-wise operations."""
    import torch

    print(f"\nElement-wise Operations ({name})")
    print("-" * 50)

    x = torch.randn(size, size).to(device)
    y = torch.randn(size, size).to(device)

    # Warmup
    for _ in range(3):
        _ = x + y
        _ = x * y
        _ = torch.sin(x)

    # Addition
    start = time.time()
    for _ in range(10):
        _ = x + y
    elapsed = time.time() - start
    add_ms = (elapsed / 10) * 1000
    print(f"  Addition: {add_ms:.2f}ms")

    # Multiplication
    start = time.time()
    for _ in range(10):
        _ = x * y
    elapsed = time.time() - start
    mul_ms = (elapsed / 10) * 1000
    print(f"  Element-wise multiply: {mul_ms:.2f}ms")

    # Sin
    start = time.time()
    for _ in range(10):
        _ = torch.sin(x)
    elapsed = time.time() - start
    sin_ms = (elapsed / 10) * 1000
    print(f"  Sin: {sin_ms:.2f}ms")

    return add_ms

def benchmark_mlp(device, name):
    """Benchmark simple MLP forward pass."""
    import torch
    import torch.nn as nn

    print(f"\nMLP Forward Pass ({name})")
    print("-" * 50)

    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    x = torch.randn(32, 1024).to(device)

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Time
    start = time.time()
    for _ in range(10):
        _ = model(x)
    elapsed = time.time() - start

    avg_ms = (elapsed / 10) * 1000
    print(f"  Batch size 32: {avg_ms:.2f}ms")

    return avg_ms

def main():
    """Run benchmarks on both CPU and DirectML."""
    import torch

    print("=" * 60)
    print("torch-directml Benchmark")
    print("=" * 60)

    try:
        import torch_directml
    except ImportError:
        print("✗ torch-directml not installed")
        print("  Run: pip install torch-directml")
        return 1

    # Check device availability
    try:
        dml_device = torch_directml.device()
    except Exception as e:
        print(f"✗ DirectML device not available: {e}")
        return 1

    sizes = [512, 1024, 2048]
    large_size = 2048

    # CPU benchmarks
    print("\n" + "=" * 60)
    print("CPU Benchmarks")
    print("=" * 60)

    cpu_matmul = benchmark_matmul("cpu", sizes, "CPU")
    cpu_elementwise = benchmark_elementwise("cpu", large_size, "CPU")
    cpu_mlp = benchmark_mlp("cpu", "CPU")

    # DirectML benchmarks
    print("\n" + "=" * 60)
    print("DirectML Benchmarks")
    print("=" * 60)

    dml_matmul = benchmark_matmul(dml_device, sizes, "DirectML")
    dml_elementwise = benchmark_elementwise(dml_device, large_size, "DirectML")
    dml_mlp = benchmark_mlp(dml_device, "DirectML")

    # Speedup comparison
    print("\n" + "=" * 60)
    print("Speedup (CPU / DirectML)")
    print("=" * 60)

    if dml_matmul > 0:
        speedup = cpu_matmul / dml_matmul
        print(f"Matrix Multiplication (2048x2048): {speedup:.2f}x")

    if dml_elementwise > 0:
        speedup = cpu_elementwise / dml_elementwise
        print(f"Element-wise operations: {speedup:.2f}x")

    if dml_mlp > 0:
        speedup = cpu_mlp / dml_mlp
        print(f"MLP Forward Pass: {speedup:.2f}x")

    print("\n✓ Benchmark complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
