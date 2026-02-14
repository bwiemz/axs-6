"""
Example: Train MNIST with AXS-6 Quantization
=============================================

Demonstrates AXS-6 mixed-precision training on MNIST digit classification.
Compares training curves and final accuracy between FP32 baseline and
AXS-6 quantized training at different block sizes.

Run: ``python -m examples.train_mnist``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.nn.modules import AXSLinear, convert_to_axs


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class MNISTNet(nn.Module):
    """Simple feedforward network for MNIST."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MNISTConvNet(nn.Module):
    """CNN for MNIST — tests Conv2d quantization."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch, return loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MNIST AXS-6 training comparison."""
    print("=" * 80)
    print("MNIST TRAINING: AXS-6 vs FP32 COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    except ImportError:
        print("\ntorchvision not installed. Generating synthetic data for demo...")
        train_dataset = _SyntheticMNIST(60000)
        test_dataset = _SyntheticMNIST(10000)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    num_epochs = 10
    lr = 1e-3

    configs = {
        "FP32 Baseline": {"block_size": None},
        "AXS-6 (B=32)": {"block_size": 32},
        "AXS-6 (B=16)": {"block_size": 16},
    }

    results: dict[str, list[dict]] = {}

    for config_name, config in configs.items():
        print(f"\n{'─' * 80}")
        print(f"Training: {config_name}")
        print(f"{'─' * 80}")

        # Create model
        torch.manual_seed(42)
        model = MNISTNet(hidden=256).to(device)

        if config["block_size"] is not None:
            model = convert_to_axs(model, block_size=config["block_size"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        epoch_results = []

        start_time = time.time()

        for epoch in range(num_epochs):
            train_stats = train_one_epoch(model, train_loader, optimizer, device)
            test_stats = evaluate(model, test_loader, device)

            epoch_results.append({
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_loss": test_stats["loss"],
                "test_acc": test_stats["accuracy"],
            })

            print(
                f"  Epoch {epoch+1:>2}/{num_epochs}: "
                f"Train Loss={train_stats['loss']:.4f}, "
                f"Train Acc={train_stats['accuracy']:.4f}, "
                f"Test Loss={test_stats['loss']:.4f}, "
                f"Test Acc={test_stats['accuracy']:.4f}"
            )

        elapsed = time.time() - start_time
        results[config_name] = epoch_results

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        fp32_bytes = num_params * 4
        if config["block_size"]:
            axs_bits = 6 + 10 / config["block_size"]
            axs_bytes = int(num_params * axs_bits / 8)
        else:
            axs_bytes = fp32_bytes

        print(f"\n  Final Test Accuracy: {epoch_results[-1]['test_acc']:.4f}")
        print(f"  Training Time: {elapsed:.1f}s")
        print(f"  Parameters: {num_params:,}")
        print(f"  Weight Memory: {axs_bytes / 1024:.1f} KB "
              f"(vs {fp32_bytes / 1024:.1f} KB FP32, "
              f"{axs_bytes / fp32_bytes:.1%})")

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<20} {'Final Test Acc':>15} {'Acc vs FP32':>15}")
    print(f"{'─' * 20} {'─' * 15} {'─' * 15}")

    fp32_acc = results["FP32 Baseline"][-1]["test_acc"]
    for name, res in results.items():
        acc = res[-1]["test_acc"]
        diff = acc - fp32_acc
        print(f"{name:<20} {acc:>14.4f}  {diff:>+14.4f}")


class _SyntheticMNIST:
    """Synthetic MNIST-like dataset for when torchvision is not available."""

    def __init__(self, size: int) -> None:
        self.data = torch.randn(size, 1, 28, 28)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


if __name__ == "__main__":
    main()
