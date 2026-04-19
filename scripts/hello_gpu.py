"""
Simple "Hello GPU" demo script for torch-directml.

Usage:
    python scripts/hello_gpu.py

Creates tensors, performs operations, and trains a simple MLP on DirectML GPU.
Good for verifying end-to-end GPU setup is working.
"""

import sys

def main():
    """Run Hello GPU demo."""
    try:
        import torch
        import torch.nn as nn
        import torch_directml
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Install dependencies: pip install -r requirements.txt")
        return 1

    print("=" * 60)
    print("torch-directml Hello GPU Demo")
    print("=" * 60)
    print()

    # Get DirectML device
    device = torch_directml.device()
    print(f"Using device: {device}")
    print()

    # Basic tensor creation
    print("1. Creating tensors on GPU...")
    x = torch.randn(5, 5).to(device)
    y = torch.randn(5, 5).to(device)
    print(f"   x shape: {x.shape}, device: {x.device}")
    print(f"   y shape: {y.shape}, device: {y.device}")
    print()

    # Matrix operations
    print("2. Performing matrix operations...")
    z = torch.matmul(x, y)
    print(f"   x @ y shape: {z.shape}")
    print(f"   Result device: {z.device}")
    print()

    # Define simple neural network
    print("3. Creating and training a neural network...")
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"   Model on device: {next(model.parameters()).device}")
    print()

    # Training loop
    print("4. Running 50 training steps...")
    print()
    print("   Epoch  | Loss      ")
    print("   " + "-" * 30)

    for epoch in range(50):
        # Generate random batch
        batch_x = torch.randn(32, 10).to(device)
        batch_y = torch.rand(32, 1).to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"   {epoch + 1:>3d}    | {loss.item():.6f}")

    print()
    print("=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    print()
    print("Your GPU setup is working correctly.")
    print("You can now run more complex models and training loops.")
    print()
    print("Buy me a coffee: https://buymeacoffee.com/chharith")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
