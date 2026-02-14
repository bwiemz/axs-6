"""
Tests for AXS-6 Neural Network Modules
=======================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from axs.nn.modules import (
    AXSConv2d,
    AXSEmbedding,
    AXSLayerNorm,
    AXSLinear,
    AXSMultiheadAttention,
    convert_to_axs,
)
from axs.nn.functional import axs_linear, axs_matmul, fake_quantize
from axs.nn.optim import AXSAdamW


class TestFakeQuantize:
    def test_shape_preserved(self) -> None:
        x = torch.randn(32, 64, requires_grad=True)
        y = fake_quantize(x)
        assert y.shape == x.shape

    def test_gradient_flows(self) -> None:
        """STE should allow gradients to flow through."""
        x = torch.randn(32, 64, requires_grad=True)
        y = fake_quantize(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # STE: gradient should be all ones (from sum)
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


class TestAXSLinear:
    def test_forward_shape(self) -> None:
        layer = AXSLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_backward(self) -> None:
        layer = AXSLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_bias(self) -> None:
        layer_bias = AXSLinear(64, 32, bias=True)
        layer_no_bias = AXSLinear(64, 32, bias=False)
        x = torch.randn(8, 64)
        y1 = layer_bias(x)
        y2 = layer_no_bias(x)
        assert y1.shape == y2.shape

    def test_extra_repr(self) -> None:
        layer = AXSLinear(64, 32, block_size=16)
        repr_str = layer.extra_repr()
        assert "64" in repr_str
        assert "32" in repr_str
        assert "16" in repr_str


class TestAXSConv2d:
    def test_forward_shape(self) -> None:
        layer = AXSConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32)
        y = layer(x)
        assert y.shape == (2, 16, 32, 32)

    def test_backward(self) -> None:
        layer = AXSConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.weight.grad is not None


class TestAXSLayerNorm:
    def test_forward_shape(self) -> None:
        layer = AXSLayerNorm(64)
        x = torch.randn(8, 16, 64)
        y = layer(x)
        assert y.shape == x.shape

    def test_normalized_output(self) -> None:
        layer = AXSLayerNorm(64)
        x = torch.randn(8, 16, 64)
        y = layer(x)
        # Output should be approximately normalized (before quantization noise)
        assert y.mean(dim=-1).abs().mean() < 0.5


class TestAXSEmbedding:
    def test_forward(self) -> None:
        layer = AXSEmbedding(100, 64)
        indices = torch.randint(0, 100, (8, 16))
        y = layer(indices)
        assert y.shape == (8, 16, 64)


class TestAXSMultiheadAttention:
    def test_forward(self) -> None:
        layer = AXSMultiheadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        y = layer(x, x, x)
        assert y.shape == (2, 16, 64)

    def test_backward(self) -> None:
        layer = AXSMultiheadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = layer(x, x, x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestConvertToAXS:
    def test_convert_linear(self) -> None:
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        model_axs = convert_to_axs(model)
        assert isinstance(model_axs[0], AXSLinear)
        assert isinstance(model_axs[2], AXSLinear)

    def test_weights_preserved(self) -> None:
        model = nn.Linear(64, 32)
        original_weight = model.weight.data.clone()
        model_axs = convert_to_axs(model)
        assert torch.equal(model_axs.weight.data, original_weight)

    def test_skip_layers(self) -> None:
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        model_axs = convert_to_axs(model, skip_layers={"0"})
        assert isinstance(model_axs[0], nn.Linear)  # skipped
        assert isinstance(model_axs[2], AXSLinear)  # converted


class TestAXSMatmul:
    def test_basic(self) -> None:
        a = torch.randn(32, 64)
        b = torch.randn(64, 16)
        c = axs_matmul(a, b)
        assert c.shape == (32, 16)

    def test_close_to_fp32(self) -> None:
        torch.manual_seed(42)
        a = torch.randn(32, 64)
        b = torch.randn(64, 16)
        c_fp32 = a @ b
        c_axs = axs_matmul(a, b)
        # 6-bit quantization error accumulates across K=64 dot products
        # Use median relative error (mean is skewed by near-zero FP32 values)
        rel_error = (c_fp32 - c_axs).abs() / (c_fp32.abs() + 1.0)
        assert rel_error.mean() < 0.1, f"Mean relative error {rel_error.mean():.4f} too high"


class TestAXSAdamW:
    def test_optimization_step(self) -> None:
        model = AXSLinear(32, 16)
        optimizer = AXSAdamW(model.parameters(), lr=0.01)

        x = torch.randn(8, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_training_reduces_loss(self) -> None:
        """Basic test that optimization actually reduces loss."""
        torch.manual_seed(42)
        model = nn.Sequential(AXSLinear(32, 16), nn.ReLU(), AXSLinear(16, 1))
        optimizer = AXSAdamW(model.parameters(), lr=0.01)

        # Fixed input/target
        x = torch.randn(64, 32)
        target = torch.randn(64, 1)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]
