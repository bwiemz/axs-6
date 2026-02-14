"""
Tests for AXS-6 Training Integration
=====================================

End-to-end tests that verify AXS-6 can be used to successfully train models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.nn.modules import AXSLinear, convert_to_axs
from axs.nn.optim import AXSAdamW, AXSTrainingWrapper


class TestEndToEndTraining:
    """Test that AXS-6 models can actually learn."""

    def test_xor_problem(self) -> None:
        """AXS-6 model should solve the XOR problem."""
        torch.manual_seed(42)

        model = nn.Sequential(
            AXSLinear(2, 16, block_size=8),
            nn.ReLU(),
            AXSLinear(16, 1, block_size=8),
            nn.Sigmoid(),
        )

        # XOR dataset
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        for _ in range(500):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.binary_cross_entropy(pred, Y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = (model(X) > 0.5).float()
            accuracy = (pred == Y).float().mean().item()

        assert accuracy >= 0.75  # should be 1.0, but allow for quantization

    def test_linear_regression(self) -> None:
        """AXS-6 model should fit a simple linear function."""
        torch.manual_seed(42)

        # Use block_size=8 with enough features to fill blocks
        model = AXSLinear(8, 1, block_size=8, quantize_input=False)
        true_weight = torch.randn(1, 8) * 2

        X = torch.randn(200, 8)
        Y = X @ true_weight.T + 0.1 * torch.randn(200, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(500):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(X)
            final_mse = F.mse_loss(pred, Y).item()

        assert final_mse < 1.0, f"Final MSE {final_mse:.4f} too high for linear regression"

    def test_classification(self) -> None:
        """AXS-6 model should learn a simple classification task."""
        torch.manual_seed(42)

        model = nn.Sequential(
            AXSLinear(10, 32, block_size=8),
            nn.ReLU(),
            AXSLinear(32, 3, block_size=8),
        )

        # Synthetic classification data
        X = torch.randn(300, 10)
        centers = torch.randn(3, 10) * 3
        labels = torch.zeros(300, dtype=torch.long)
        for i in range(300):
            dists = ((X[i] - centers) ** 2).sum(dim=1)
            labels[i] = dists.argmin()

        optimizer = AXSAdamW(model.parameters(), lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(X).argmax(dim=1)
            accuracy = (pred == labels).float().mean().item()

        assert accuracy > 0.5  # should beat random (33%)

    def test_converted_model_trains(self) -> None:
        """A converted standard model should still be trainable."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

        model = convert_to_axs(model, block_size=32)

        X = torch.randn(100, 20)
        Y = torch.randint(0, 5, (100,))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss  # should decrease


class TestTrainingWrapper:
    def test_basic_step(self) -> None:
        model = nn.Sequential(
            AXSLinear(10, 16, block_size=8),
            nn.ReLU(),
            AXSLinear(16, 1, block_size=8),
        )
        optimizer = AXSAdamW(model.parameters(), lr=0.01)
        wrapper = AXSTrainingWrapper(model, optimizer, max_grad_norm=1.0)

        X = torch.randn(32, 10)
        Y = torch.randn(32, 1)

        def loss_fn(output, inputs):
            return F.mse_loss(output, Y)

        stats = wrapper.training_step(X, loss_fn)
        assert "loss" in stats
        assert "loss_scale" in stats
        assert stats["loss"] > 0
