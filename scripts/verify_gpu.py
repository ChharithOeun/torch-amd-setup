"""
Verify torch-directml GPU setup and functionality.

Usage:
    python scripts/verify_gpu.py

This script checks that PyTorch, torch-directml are installed and the GPU is accessible.
It runs basic tensor operations to verify DirectML backend is working correctly.
"""

import sys
import traceback

def check_torch_installation():
    """Check if torch is installed and report version."""
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Fix: pip install torch")
        return False

def check_torch_directml():
    """Check if torch-directml is installed."""
    try:
        import torch_directml
        print(f"✓ torch-directml available")
        return True
    except ImportError:
        print("✗ torch-directml not installed")
        print("  Fix: pip install torch-directml")
        return False

def check_device_count():
    """Get DirectML device count."""
    try:
        import torch_directml
        device_count = torch_directml.device_count()
        print(f"✓ DirectML device count: {device_count}")
        return device_count > 0
    except Exception as e:
        print(f"✗ Failed to get device count: {e}")
        return False

def check_device_name():
    """Get DirectML device name."""
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"✓ DirectML device: {device}")
        return True
    except Exception as e:
        print(f"✗ Failed to get device name: {e}")
        return False

def test_matmul_operation():
    """Test matrix multiplication on DirectML device."""
    try:
        import torch
        import torch_directml

        device = torch_directml.device()

        # Create test tensors
        a = torch.randn(512, 512).to(device)
        b = torch.randn(512, 512).to(device)

        # Time the operation
        import time
        start = time.time()
        c = torch.matmul(a, b)
        elapsed = time.time() - start

        print(f"✓ Matrix multiplication (512x512): {elapsed*1000:.2f}ms")
        return True
    except Exception as e:
        print(f"✗ Matrix multiplication failed: {e}")
        traceback.print_exc()
        return False

def test_model_forward_pass():
    """Test forward pass of a simple neural network on DirectML."""
    try:
        import torch
        import torch.nn as nn
        import torch_directml

        device = torch_directml.device()

        # Simple linear layer
        model = nn.Linear(128, 64).to(device)
        x = torch.randn(32, 128).to(device)

        import time
        start = time.time()
        y = model(x)
        elapsed = time.time() - start

        print(f"✓ Neural network forward pass: {elapsed*1000:.2f}ms")
        return True
    except Exception as e:
        print(f"✗ Neural network forward pass failed: {e}")
        traceback.print_exc()
        return False

def estimate_vram():
    """Estimate available VRAM."""
    try:
        import torch
        import torch_directml

        device = torch_directml.device()

        # Try allocating progressively larger tensors
        size_mb = 64
        max_mb = 0

        while size_mb <= 8192:
            try:
                tensor = torch.zeros(size_mb * 1024 * 1024 // 4, dtype=torch.float32).to(device)
                max_mb = size_mb
                size_mb *= 2
            except RuntimeError:
                break

        if max_mb > 0:
            print(f"✓ Estimated available VRAM: ~{max_mb}MB+")
            return True
        else:
            print("⚠ Could not estimate VRAM")
            return True
    except Exception as e:
        print(f"⚠ VRAM estimation skipped: {e}")
        return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("torch-amd-setup GPU Verification")
    print("=" * 60)
    print()

    checks = [
        ("PyTorch", check_torch_installation),
        ("torch-directml", check_torch_directml),
        ("Device Count", check_device_count),
        ("Device Name", check_device_name),
        ("Matrix Multiplication", test_matmul_operation),
        ("Neural Network", test_model_forward_pass),
        ("VRAM Estimation", estimate_vram),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed: {e}")
            traceback.print_exc()
            results.append((name, False))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ GPU setup verified! You can now run training scripts.")
        return 0
    else:
        print("\n✗ Some checks failed. See above for fix suggestions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
