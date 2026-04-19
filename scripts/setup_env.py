"""
Environment setup wizard for torch-directml.

Usage:
    python scripts/setup_env.py
    python scripts/setup_env.py --dry-run
    python scripts/setup_env.py --silent

Checks Python version, installed packages, and installs torch-directml and dependencies.
Verifies setup after installation.
"""

import sys
import subprocess
import argparse

def check_python_version():
    """Verify Python 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"✗ Python {version.major}.{version.minor} detected")
        print("  torch-directml requires Python 3.10 or higher")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_torch_installation():
    """Check if torch is already installed."""
    try:
        import torch
        version = torch.__version__
        print(f"✓ PyTorch {version} already installed")

        # Warn if CUDA version
        if "cu" in version.lower():
            print("  ⚠ Warning: CUDA version of PyTorch detected")
            print("  torch-directml may conflict with CUDA PyTorch")
            print("  Consider reinstalling with: pip install torch-directml")
            return True

        return True
    except ImportError:
        print("• PyTorch not installed yet")
        return False

def check_torch_directml():
    """Check if torch-directml is installed."""
    try:
        import torch_directml
        print("✓ torch-directml already installed")
        return True
    except ImportError:
        print("• torch-directml not installed yet")
        return False

def install_package(package, dry_run=False, silent=False):
    """Install a package via pip."""
    if not silent:
        print(f"  Installing {package}...")

    if dry_run:
        if not silent:
            print(f"  [DRY-RUN] Would run: pip install {package}")
        return True

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL if silent else None,
            stderr=subprocess.DEVNULL if silent else None
        )
        if not silent:
            print(f"  ✓ {package} installed")
        return True
    except subprocess.CalledProcessError:
        if not silent:
            print(f"  ✗ Failed to install {package}")
        return False

def install_dependencies(dry_run=False, silent=False):
    """Install torch-directml and related packages."""
    packages = [
        "torch-directml",
        "diffusers",
        "transformers",
        "accelerate"
    ]

    if not silent:
        print("\nInstalling packages:")

    results = []
    for package in packages:
        success = install_package(package, dry_run=dry_run, silent=silent)
        results.append((package, success))

    return all(success for _, success in results)

def run_verification(silent=False):
    """Run verification script after installation."""
    if not silent:
        print("\nRunning verification...")

    try:
        import torch
        import torch_directml

        device = torch_directml.device()
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)

        if not silent:
            print("✓ Verification passed")
        return True
    except Exception as e:
        if not silent:
            print(f"✗ Verification failed: {e}")
        return False

def main():
    """Run setup wizard."""
    parser = argparse.ArgumentParser(description="Setup torch-directml environment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed")
    parser.add_argument("--silent", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not args.silent:
        print("=" * 60)
        print("torch-directml Setup Wizard")
        print("=" * 60)
        print()

    # Check Python version
    if not args.silent:
        print("Checking environment:")
    if not check_python_version():
        return 1

    # Check existing installations
    check_torch_installation()
    check_torch_directml()

    # Install packages
    if not args.silent:
        print()
    if not install_dependencies(dry_run=args.dry_run, silent=args.silent):
        if not args.silent:
            print("\n✗ Some packages failed to install")
        return 1

    # Verify
    if not args.dry_run:
        if not run_verification(silent=args.silent):
            if not args.silent:
                print("  Run: python scripts/verify_gpu.py")
            return 1

    if not args.silent:
        print()
        print("=" * 60)
        if args.dry_run:
            print("Setup preview complete (dry-run)")
        else:
            print("✓ Setup complete!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("  1. Verify GPU setup: python scripts/verify_gpu.py")
            print("  2. Run demo: python scripts/hello_gpu.py")
            print("  3. Check benchmarks: python scripts/benchmark.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())
