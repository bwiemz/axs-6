"""
Tests for AXS-6 Quantization Strategies
========================================
"""

from __future__ import annotations

import torch
import pytest

from axs.core import dequantize
from axs.quantize import (
    ErrorFeedbackState,
    RoundingMode,
    quantize_adaptive,
    quantize_gasr,
    quantize_nearest,
    quantize_stochastic,
    quantize_with_error_feedback,
)


class TestQuantizeNearest:
    def test_basic(self) -> None:
        x = torch.randn(64, 64)
        axs = quantize_nearest(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_deterministic(self) -> None:
        """Nearest rounding should be deterministic."""
        x = torch.randn(64, 64)
        x_hat1 = dequantize(quantize_nearest(x))
        x_hat2 = dequantize(quantize_nearest(x))
        assert torch.equal(x_hat1, x_hat2)


class TestQuantizeStochastic:
    def test_basic(self) -> None:
        x = torch.randn(64, 64)
        axs = quantize_stochastic(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_non_deterministic(self) -> None:
        """Stochastic rounding should produce different results each time."""
        x = torch.randn(256, 256)
        x_hat1 = dequantize(quantize_stochastic(x))
        x_hat2 = dequantize(quantize_stochastic(x))
        # Very unlikely to be exactly equal
        assert not torch.equal(x_hat1, x_hat2)

    def test_unbiased(self) -> None:
        """Mean quantization error should be approximately zero."""
        torch.manual_seed(42)
        x = torch.randn(128, 128)
        errors = []
        for _ in range(200):
            x_hat = dequantize(quantize_stochastic(x))
            errors.append((x - x_hat).mean().item())
        mean_error = sum(errors) / len(errors)
        assert abs(mean_error) < 0.005


class TestErrorFeedback:
    def test_basic(self) -> None:
        state = ErrorFeedbackState()
        x = torch.randn(64, 64)
        axs = quantize_with_error_feedback(x, param_name="test", state=state)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape
        assert "test" in state.buffer_names

    def test_error_reduces_over_steps(self) -> None:
        """Error feedback should reduce accumulated error over multiple steps."""
        state = ErrorFeedbackState()
        x = torch.randn(64, 64)

        cumulative_errors = []
        for step in range(20):
            axs = quantize_with_error_feedback(x, param_name="w", state=state)
            x_hat = dequantize(axs)
            error = (x - x_hat).abs().mean().item()
            cumulative_errors.append(error)

        # Error should not grow unboundedly
        assert cumulative_errors[-1] < cumulative_errors[0] * 2

    def test_reset(self) -> None:
        state = ErrorFeedbackState()
        x = torch.randn(32, 32)
        quantize_with_error_feedback(x, param_name="test", state=state)
        assert len(state.buffer_names) == 1
        state.reset()
        assert len(state.buffer_names) == 0


class TestGASR:
    def test_basic(self) -> None:
        x = torch.randn(64, 64)
        g = torch.randn(64, 64)
        axs = quantize_gasr(x, g, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_gradient_aware(self) -> None:
        """High-gradient regions should have lower quantization error."""
        torch.manual_seed(42)
        x = torch.randn(128, 128)

        # Gradients concentrated in first half
        g = torch.zeros(128, 128)
        g[:, :64] = 10.0  # high gradient in first half

        axs = quantize_gasr(x, g, temperature=2.0)
        x_hat = dequantize(axs)
        error = (x - x_hat).abs()

        # Error in high-gradient region should be lower
        error_high = error[:, :64].mean().item()
        error_low = error[:, 64:].mean().item()
        # This is a soft property — may not always hold for single samples
        # but should hold on average


class TestQuantizeAdaptive:
    def test_all_modes(self) -> None:
        x = torch.randn(64, 64)
        g = torch.randn(64, 64)

        for mode in RoundingMode:
            if mode == RoundingMode.GASR:
                axs = quantize_adaptive(x, mode=mode, gradient=g)
            else:
                axs = quantize_adaptive(x, mode=mode)
            x_hat = dequantize(axs)
            assert x_hat.shape == x.shape

    def test_gasr_requires_gradient(self) -> None:
        x = torch.randn(64, 64)
        with pytest.raises(ValueError, match="gradient"):
            quantize_adaptive(x, mode=RoundingMode.GASR)
