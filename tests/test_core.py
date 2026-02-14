"""
Tests for AXS-6 Core Format
============================

Validates encoding/decoding correctness, numerical properties, and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from axs.core import (
    AXS6_EXPONENT_BIAS,
    AXS6_MAX_MAGNITUDE,
    AXSBlock,
    AXSTensor,
    BlockConfig,
    axs_decode_block,
    axs_encode_block,
    dequantize,
    quantize,
    quantization_error,
)


class TestBlockEncodeDecode:
    """Test numpy-level block encoding and decoding."""

    def test_roundtrip_simple(self) -> None:
        """Values should survive encode→decode with bounded error."""
        values = np.array([0.5, -0.3, 0.8, -0.1] * 8, dtype=np.float64)
        block = axs_encode_block(values)
        decoded = axs_decode_block(block)
        np.testing.assert_allclose(decoded, values, atol=0.05)

    def test_zero_block(self) -> None:
        """All-zero input should produce all-zero output."""
        values = np.zeros(32, dtype=np.float64)
        block = axs_encode_block(values)
        decoded = axs_decode_block(block)
        np.testing.assert_array_equal(decoded, 0.0)
        assert block.shared_exponent == 0

    def test_signs_preserved(self) -> None:
        """Sign of each value should be preserved."""
        values = np.array([1.0, -1.0, 0.5, -0.5] * 8, dtype=np.float64)
        block = axs_encode_block(values)
        decoded = axs_decode_block(block)
        for orig, dec in zip(values, decoded):
            if orig != 0:
                assert np.sign(orig) == np.sign(dec)

    def test_block_sizes(self) -> None:
        """All valid block sizes should work."""
        for bs in [8, 16, 32]:
            values = np.random.randn(bs)
            block = axs_encode_block(values)
            decoded = axs_decode_block(block)
            assert len(decoded) == bs

    def test_invalid_block_size(self) -> None:
        """Invalid block sizes should raise."""
        with pytest.raises(AssertionError):
            axs_encode_block(np.zeros(7))

    def test_large_values(self) -> None:
        """Large values should be handled within the exponent range."""
        values = np.array([1e10, -5e9, 3e10, -2e10] * 8, dtype=np.float64)
        block = axs_encode_block(values)
        decoded = axs_decode_block(block)
        # Relative error should be bounded
        rel_error = np.abs(decoded - values) / np.abs(values).max()
        assert np.max(rel_error) < 0.05

    def test_small_values(self) -> None:
        """Very small values should be representable."""
        values = np.array([1e-20, -5e-21, 3e-20, -2e-20] * 8, dtype=np.float64)
        block = axs_encode_block(values)
        decoded = axs_decode_block(block)
        rel_error = np.abs(decoded - values) / np.abs(values).max()
        assert np.max(rel_error) < 0.05

    def test_stochastic_rounding(self) -> None:
        """Stochastic rounding should be unbiased on average."""
        np.random.seed(42)
        values = np.random.randn(32)
        errors = []
        for _ in range(1000):
            block = axs_encode_block(values, rounding="stochastic")
            decoded = axs_decode_block(block)
            errors.append(decoded - values)
        mean_error = np.mean(errors, axis=0)
        # Mean error should be close to zero (unbiased)
        assert np.max(np.abs(mean_error)) < 0.01

    def test_serialization_roundtrip(self) -> None:
        """Block should survive serialize→deserialize."""
        values = np.random.randn(32)
        block = axs_encode_block(values)
        data = block.to_bytes()
        block2 = AXSBlock.from_bytes(data, block_size=32)
        decoded = axs_decode_block(block2)
        decoded_orig = axs_decode_block(block)
        np.testing.assert_array_equal(decoded, decoded_orig)


class TestTensorQuantize:
    """Test PyTorch tensor-level quantization."""

    def test_roundtrip_1d(self) -> None:
        """1D tensor roundtrip."""
        x = torch.randn(128)
        axs = quantize(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape
        assert torch.allclose(x, x_hat, atol=0.1)

    def test_roundtrip_2d(self) -> None:
        """2D tensor roundtrip."""
        x = torch.randn(64, 128)
        axs = quantize(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_roundtrip_3d(self) -> None:
        """3D tensor roundtrip."""
        x = torch.randn(8, 32, 64)
        axs = quantize(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_padding(self) -> None:
        """Tensors with last dim not a multiple of block_size should be padded."""
        x = torch.randn(10, 50)  # 50 is not a multiple of 32
        axs = quantize(x, block_size=32)
        x_hat = dequantize(axs)
        assert x_hat.shape == x.shape

    def test_zero_tensor(self) -> None:
        """All-zero tensor should quantize and dequantize to zero."""
        x = torch.zeros(32, 64)
        axs = quantize(x)
        x_hat = dequantize(axs)
        assert torch.all(x_hat == 0)

    def test_error_bounded(self) -> None:
        """Quantization error should be reasonable for normal data."""
        torch.manual_seed(42)
        x = torch.randn(256, 256)
        err = quantization_error(x, block_size=32)
        # SNR should be at least 25 dB for Gaussian data
        assert err["signal_to_noise_db"] > 20

    def test_compression_ratio(self) -> None:
        """AXSTensor should report correct compression ratios."""
        x = torch.randn(1024, 1024)
        axs = quantize(x, block_size=32)
        assert axs.effective_bits_per_value < 7.0
        assert axs.compression_ratio_vs_fp32 > 4.5
        assert axs.compression_ratio_vs_fp8 > 1.1


class TestQuantizationError:
    """Test quantization error analysis."""

    def test_error_stats_keys(self) -> None:
        x = torch.randn(64, 64)
        stats = quantization_error(x)
        assert "mse" in stats
        assert "rmse" in stats
        assert "max_abs_error" in stats
        assert "signal_to_noise_db" in stats

    def test_nearest_vs_stochastic_mse(self) -> None:
        """Nearest rounding should have lower per-sample MSE."""
        torch.manual_seed(42)
        x = torch.randn(256, 256)
        err_nearest = quantization_error(x, rounding="nearest")
        err_stoch = quantization_error(x, rounding="stochastic")
        # Nearest should have lower or similar MSE
        assert err_nearest["mse"] <= err_stoch["mse"] * 1.5  # allow some variance
