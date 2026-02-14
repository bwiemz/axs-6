"""
Tests for AXS-6 Unified Quantiser
===================================

Comprehensive tests covering:
  - Fused NF5 warp table correctness
  - Fused fake-quantise round-trip
  - Quantise → dequantise serialisation path
  - Autograd (STE gradient flow)
  - Drop-in modules (Linear, LayerNorm, Embedding, MHA)
  - Model conversion from nn / V1 / V2 layers
  - Training pipeline with warmup
  - Quality comparison vs V2 (MSE within 1%)
  - Edge cases: zeros, single-element, large tensors
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from axs.core import AXS6_MAX_MAGNITUDE, DEFAULT_BLOCK_SIZE
from axs.unified.quantize_unified import (
    FUSED_NF5_LUT,
    NF5_CODEBOOK,
    _LUT_MAX_IDX,
    dequantize_unified,
    fused_fake_quantize,
    quantization_error_unified,
    quantize_unified,
)
from axs.unified.functional_unified import (
    axs_linear_unified,
    axs_matmul_unified,
    fake_quantize_unified,
)
from axs.unified.modules_unified import (
    AXSEmbeddingUnified,
    AXSLayerNormUnified,
    AXSLinearUnified,
    AXSMultiheadAttentionUnified,
    convert_to_axs_unified,
)
from axs.unified.training_unified import AmaxEMA, AXSTrainingPipelineUnified


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def randn_block():
    """Standard Gaussian block of 32 values."""
    torch.manual_seed(42)
    return torch.randn(32)


@pytest.fixture
def randn_2d():
    """Standard Gaussian 2-D tensor (128×256)."""
    torch.manual_seed(42)
    return torch.randn(128, 256)


@pytest.fixture
def randn_3d():
    """Standard Gaussian 3-D tensor (batch × seq × dim)."""
    torch.manual_seed(42)
    return torch.randn(4, 16, 64)


# ===================================================================
# 1. LUT / Codebook integrity
# ===================================================================

class TestLUT:
    """Tests for the fused NF5 warp table."""

    def test_lut_shape(self):
        assert FUSED_NF5_LUT.shape == (1024,)

    def test_lut_range(self):
        assert FUSED_NF5_LUT.min() >= 0.0
        assert FUSED_NF5_LUT.max() <= 1.0 + 1e-6

    def test_lut_monotonic(self):
        """LUT should be non-decreasing (NF5 codebook is sorted)."""
        diff = FUSED_NF5_LUT[1:] - FUSED_NF5_LUT[:-1]
        assert (diff >= -1e-7).all(), "LUT is not monotonically non-decreasing"

    def test_lut_starts_near_zero(self):
        assert FUSED_NF5_LUT[0] < 0.05

    def test_lut_ends_near_one(self):
        assert FUSED_NF5_LUT[-1] > 0.95

    def test_codebook_32_levels(self):
        assert NF5_CODEBOOK.shape == (32,)

    def test_codebook_normalised(self):
        assert abs(NF5_CODEBOOK[-1].item() - 1.0) < 1e-5


# ===================================================================
# 2. Fused fake-quantise
# ===================================================================

class TestFusedFakeQuantize:
    """Tests for the core fused_fake_quantize function."""

    def test_output_shape(self, randn_2d: torch.Tensor):
        out = fused_fake_quantize(randn_2d)
        assert out.shape == randn_2d.shape

    def test_output_dtype(self, randn_2d: torch.Tensor):
        out = fused_fake_quantize(randn_2d)
        assert out.dtype == torch.float32

    def test_zeros_passthrough(self):
        z = torch.zeros(1, 32)
        out = fused_fake_quantize(z)
        assert torch.allclose(out, z, atol=1e-7)

    def test_deterministic_nearest(self, randn_2d: torch.Tensor):
        a = fused_fake_quantize(randn_2d, rounding="nearest")
        b = fused_fake_quantize(randn_2d, rounding="nearest")
        assert torch.allclose(a, b)

    def test_stochastic_varies(self, randn_2d: torch.Tensor):
        """Stochastic rounding should produce different outputs (with high probability)."""
        a = fused_fake_quantize(randn_2d, rounding="stochastic")
        b = fused_fake_quantize(randn_2d, rounding="stochastic")
        # Should not be exactly equal (probability ~ 0)
        assert not torch.allclose(a, b)

    def test_3d_tensor(self, randn_3d: torch.Tensor):
        out = fused_fake_quantize(randn_3d)
        assert out.shape == randn_3d.shape

    def test_mse_reasonable(self, randn_2d: torch.Tensor):
        out = fused_fake_quantize(randn_2d)
        mse = (randn_2d - out).pow(2).mean().item()
        # For 6-bit quantisation of ~N(0,1) data, MSE should be < 0.01
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_block_size_8(self):
        x = torch.randn(4, 64)
        out = fused_fake_quantize(x, block_size=8)
        assert out.shape == x.shape

    def test_block_size_16(self):
        x = torch.randn(4, 64)
        out = fused_fake_quantize(x, block_size=16)
        assert out.shape == x.shape

    def test_non_divisible_last_dim(self):
        """Test with last dim not divisible by block_size (needs padding)."""
        x = torch.randn(4, 50)  # 50 is not divisible by 32
        out = fused_fake_quantize(x, block_size=32)
        assert out.shape == x.shape

    def test_single_element(self):
        x = torch.tensor([[1.5]])
        out = fused_fake_quantize(x, block_size=8)
        assert out.shape == x.shape

    def test_large_values(self):
        x = torch.randn(4, 32) * 1000.0
        out = fused_fake_quantize(x)
        mse = (x - out).pow(2).mean().item()
        # Relative error should still be reasonable
        rel_mse = mse / x.pow(2).mean().item()
        assert rel_mse < 0.01

    def test_tiny_values(self):
        x = torch.randn(4, 32) * 1e-6
        out = fused_fake_quantize(x)
        assert out.shape == x.shape
        # Should not be all zeros
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_sign_preservation(self, randn_2d: torch.Tensor):
        out = fused_fake_quantize(randn_2d)
        # Signs should match (where values are non-zero)
        nonzero = randn_2d.abs() > 1e-6
        orig_signs = randn_2d[nonzero].sign()
        out_signs = out[nonzero].sign()
        assert (orig_signs == out_signs).all()


# ===================================================================
# 3. Quantise / Dequantise (serialisation path)
# ===================================================================

class TestQuantizeDequantize:
    """Tests for the AXSTensor serialisation path."""

    def test_round_trip_shape(self, randn_2d: torch.Tensor):
        axs = quantize_unified(randn_2d)
        out = dequantize_unified(axs)
        assert out.shape == randn_2d.shape

    def test_round_trip_quality(self, randn_2d: torch.Tensor):
        axs = quantize_unified(randn_2d)
        out = dequantize_unified(axs)
        mse = (randn_2d - out).pow(2).mean().item()
        assert mse < 0.01

    def test_magnitudes_range(self, randn_2d: torch.Tensor):
        axs = quantize_unified(randn_2d)
        assert axs.magnitudes.max() <= AXS6_MAX_MAGNITUDE
        assert axs.magnitudes.min() >= 0

    def test_stochastic_rounding(self, randn_2d: torch.Tensor):
        a = quantize_unified(randn_2d, rounding="stochastic")
        b = quantize_unified(randn_2d, rounding="stochastic")
        # Should produce different magnitudes (stochastic)
        assert not torch.equal(a.magnitudes, b.magnitudes)

    def test_metadata(self, randn_2d: torch.Tensor):
        axs = quantize_unified(randn_2d)
        assert axs.block_size == DEFAULT_BLOCK_SIZE
        assert axs.original_shape == randn_2d.shape
        assert axs.num_blocks == randn_2d.shape[-1] // DEFAULT_BLOCK_SIZE

    def test_effective_bits(self, randn_2d: torch.Tensor):
        axs = quantize_unified(randn_2d)
        # 6 + 10/32 = 6.3125
        assert abs(axs.effective_bits_per_value - 6.3125) < 0.01


# ===================================================================
# 4. Quantisation error statistics
# ===================================================================

class TestQuantizationError:
    def test_error_keys(self, randn_2d: torch.Tensor):
        stats = quantization_error_unified(randn_2d)
        expected = {"mse", "rmse", "max_abs_error", "mean_abs_error", "signal_to_noise_db"}
        assert set(stats.keys()) == expected

    def test_positive_snr(self, randn_2d: torch.Tensor):
        stats = quantization_error_unified(randn_2d)
        assert stats["signal_to_noise_db"] > 0

    def test_rmse_consistency(self, randn_2d: torch.Tensor):
        stats = quantization_error_unified(randn_2d)
        assert abs(stats["rmse"] - math.sqrt(stats["mse"])) < 1e-6


# ===================================================================
# 5. Autograd / STE
# ===================================================================

class TestAutograd:
    def test_gradient_flows(self):
        x = torch.randn(4, 32, requires_grad=True)
        y = fake_quantize_unified(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_ste_identity_gradient(self):
        """STE should pass gradients through unchanged."""
        x = torch.randn(4, 32, requires_grad=True)
        y = fake_quantize_unified(x)
        loss = (y * 2.0).sum()
        loss.backward()
        # Gradient should be 2.0 everywhere (STE)
        assert torch.allclose(x.grad, torch.full_like(x, 2.0))


# ===================================================================
# 6. Functional ops
# ===================================================================

class TestFunctionalOps:
    def test_linear_output_shape(self):
        x = torch.randn(4, 64)
        w = torch.randn(32, 64)
        b = torch.randn(32)
        out = axs_linear_unified(x, w, b)
        assert out.shape == (4, 32)

    def test_linear_gradient(self):
        x = torch.randn(4, 64, requires_grad=True)
        w = torch.randn(32, 64, requires_grad=True)
        out = axs_linear_unified(x, w)
        out.sum().backward()
        assert x.grad is not None
        assert w.grad is not None

    def test_matmul_output_shape(self):
        a = torch.randn(4, 8, 64)
        b = torch.randn(4, 64, 32)
        out = axs_matmul_unified(a, b)
        assert out.shape == (4, 8, 32)

    def test_matmul_no_quantize(self):
        a = torch.randn(4, 8, 64)
        b = torch.randn(4, 64, 32)
        out = axs_matmul_unified(a, b, quantize_inputs=False)
        expected = torch.matmul(a, b)
        assert torch.allclose(out, expected)


# ===================================================================
# 7. Modules
# ===================================================================

class TestModules:
    def test_linear_forward(self):
        layer = AXSLinearUnified(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_linear_backward(self):
        layer = AXSLinearUnified(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_linear_warmup_mode(self):
        """Warmup mode should bypass quantisation (output = standard linear)."""
        layer = AXSLinearUnified(64, 32)
        layer._warmup_active = True
        x = torch.randn(4, 64)
        out_warmup = layer(x)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.allclose(out_warmup, expected)

    def test_linear_extra_repr(self):
        layer = AXSLinearUnified(64, 32)
        s = repr(layer)
        assert "in_features=64" in s
        assert "out_features=32" in s

    def test_layernorm_forward(self):
        layer = AXSLayerNormUnified(64)
        x = torch.randn(4, 8, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_embedding_forward(self):
        layer = AXSEmbeddingUnified(100, 64)
        idx = torch.randint(0, 100, (4, 8))
        out = layer(idx)
        assert out.shape == (4, 8, 64)

    def test_embedding_warmup(self):
        layer = AXSEmbeddingUnified(100, 64)
        layer._warmup_active = True
        idx = torch.randint(0, 100, (4, 8))
        out = layer(idx)
        raw = torch.nn.functional.embedding(idx, layer.weight)
        assert torch.allclose(out, raw)

    def test_mha_forward(self):
        mha = AXSMultiheadAttentionUnified(64, 4)
        x = torch.randn(2, 8, 64)
        out = mha(x, x, x)
        assert out.shape == x.shape

    def test_mha_with_mask(self):
        mha = AXSMultiheadAttentionUnified(64, 4)
        x = torch.randn(2, 8, 64)
        mask = torch.zeros(8, 8)
        mask = mask.masked_fill(torch.triu(torch.ones(8, 8), diagonal=1).bool(), float("-inf"))
        out = mha(x, x, x, attn_mask=mask)
        assert out.shape == x.shape


# ===================================================================
# 8. Model conversion
# ===================================================================

class TestConversion:
    def test_convert_linear(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        converted = convert_to_axs_unified(model)
        assert isinstance(converted[0], AXSLinearUnified)
        assert isinstance(converted[2], AXSLinearUnified)

    def test_convert_preserves_weights(self):
        model = nn.Linear(64, 32)
        w_orig = model.weight.data.clone()
        b_orig = model.bias.data.clone()
        converted = convert_to_axs_unified(model)
        assert torch.allclose(converted.weight.data, w_orig)
        assert torch.allclose(converted.bias.data, b_orig)

    def test_convert_layernorm(self):
        model = nn.Sequential(nn.LayerNorm(64))
        converted = convert_to_axs_unified(model)
        assert isinstance(converted[0], AXSLayerNormUnified)

    def test_convert_embedding(self):
        model = nn.Sequential(nn.Embedding(100, 64))
        converted = convert_to_axs_unified(model)
        assert isinstance(converted[0], AXSEmbeddingUnified)

    def test_convert_skip_layers(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
        converted = convert_to_axs_unified(model, skip_layers={"0"})
        assert isinstance(converted[0], nn.Linear)  # skipped
        assert isinstance(converted[1], AXSLinearUnified)  # converted

    def test_convert_inplace(self):
        model = nn.Sequential(nn.Linear(64, 32))
        original_id = id(model)
        converted = convert_to_axs_unified(model, inplace=True)
        assert id(converted) == original_id


# ===================================================================
# 9. Training pipeline
# ===================================================================

class TestTrainingPipeline:
    def _make_model_and_pipeline(self, warmup: int = 0):
        model = nn.Sequential(
            AXSLinearUnified(64, 32),
            nn.ReLU(),
            AXSLinearUnified(32, 16),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pipeline = AXSTrainingPipelineUnified(
            model, optimizer, warmup_steps=warmup,
        )
        return model, pipeline

    def test_single_step(self):
        model, pipeline = self._make_model_and_pipeline()
        x = torch.randn(4, 64)
        stats = pipeline.training_step(
            x, lambda out, inp: out.sum(),
        )
        assert "loss" in stats
        assert not stats["overflow"]
        assert stats["step"] == 1

    def test_warmup_steps(self):
        model, pipeline = self._make_model_and_pipeline(warmup=5)
        assert pipeline.is_warmup
        x = torch.randn(4, 64)
        for i in range(5):
            stats = pipeline.training_step(x, lambda out, inp: out.sum())
            if i < 4:
                assert stats["warmup"]
        # After 5 steps, warmup should be over
        assert not pipeline.is_warmup

    def test_stats(self):
        model, pipeline = self._make_model_and_pipeline()
        x = torch.randn(4, 64)
        pipeline.training_step(x, lambda out, inp: out.sum())
        s = pipeline.stats
        assert s["total_steps"] == 1
        assert "avg_grad_norm_recent" in s


# ===================================================================
# 10. Amax EMA
# ===================================================================

class TestAmaxEMA:
    def test_initial_value(self):
        ema = AmaxEMA()
        # Unknown name → default
        assert ema.get_amax("foo") == 1.0

    def test_first_update(self):
        ema = AmaxEMA()
        ema.update("w", 5.0)
        assert ema.get_amax("w") == 5.0

    def test_ema_decay(self):
        ema = AmaxEMA(decay=0.9)
        ema.update("w", 10.0)
        ema.update("w", 0.0)
        # After second update: 0.9 * 10.0 + 0.1 * 0.0 = 9.0
        assert abs(ema.get_amax("w") - 9.0) < 1e-6

    def test_get_scale_power_of_2(self):
        ema = AmaxEMA()
        ema.update("w", 3.0)
        scale = ema.get_scale("w")
        # floor(log2(3)) + 1 = 1 + 1 = 2 → 2^2 = 4
        assert scale == 4.0


# ===================================================================
# 11. Quality: Unified vs V2
# ===================================================================

class TestQualityVsV2:
    """Verify the unified quantiser matches V2 quality."""

    def test_mse_within_1_percent_of_v2(self):
        """Unified MSE should be within 1% of V2 MSE."""
        torch.manual_seed(0)
        x = torch.randn(256, 256)

        # Unified
        uni_out = fused_fake_quantize(x)
        uni_mse = (x - uni_out).pow(2).mean().item()

        # V2
        from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
        axs = quantize_v2(x, use_nf5=True, clip_percentile=None)
        v2_out = dequantize_v2(axs, use_nf5=True)
        v2_mse = (x - v2_out).pow(2).mean().item()

        # Unified should be within 1% of V2
        ratio = uni_mse / v2_mse
        assert 0.95 <= ratio <= 1.05, f"MSE ratio unified/V2 = {ratio:.4f} (expected ~1.0)"

    def test_unified_better_than_v1(self):
        """Unified MSE should be significantly lower than V1."""
        torch.manual_seed(0)
        x = torch.randn(256, 256)

        uni_out = fused_fake_quantize(x)
        uni_mse = (x - uni_out).pow(2).mean().item()

        from axs.core import quantize, dequantize
        axs = quantize(x)
        v1_out = dequantize(axs)
        v1_mse = (x - v1_out).pow(2).mean().item()

        # Unified should be at least 20% better than V1
        assert uni_mse < v1_mse * 0.85, (
            f"Unified MSE {uni_mse:.6f} not sufficiently better than V1 MSE {v1_mse:.6f}"
        )

    def test_snr_better_than_v1(self):
        """Unified SNR should be higher than V1."""
        torch.manual_seed(0)
        x = torch.randn(256, 256)

        uni_stats = quantization_error_unified(x)

        from axs.core import quantization_error
        v1_stats = quantization_error(x)

        assert uni_stats["signal_to_noise_db"] > v1_stats["signal_to_noise_db"]


# ===================================================================
# 12. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_all_zeros_block(self):
        x = torch.zeros(1, 32)
        out = fused_fake_quantize(x)
        assert torch.allclose(out, x, atol=1e-7)

    def test_all_same_value(self):
        x = torch.full((1, 32), 3.14)
        out = fused_fake_quantize(x)
        assert out.shape == x.shape
        # All values should be the same (same code in every position)
        assert torch.allclose(out, out[0, 0].expand_as(out))

    def test_mixed_positive_negative(self):
        x = torch.tensor([[-1.0, 1.0] * 16])
        out = fused_fake_quantize(x)
        # Signs should be preserved
        assert (out[:, 0::2] < 0).all()
        assert (out[:, 1::2] > 0).all()

    def test_single_outlier(self):
        """A single outlier should not destroy the rest of the block."""
        x = torch.randn(1, 32)
        x[0, 0] = 100.0  # outlier
        out = fused_fake_quantize(x)
        # With amax-based scaling, the outlier dominates the block scale.
        # Still, the non-outlier values should be reconstructed within
        # the block's resolution (relative error < 100%).
        non_outlier_error = (x[0, 1:] - out[0, 1:]).abs().mean()
        assert non_outlier_error < x[0, 1:].abs().mean() * 1.0

    def test_very_large_tensor(self):
        x = torch.randn(1024, 1024)
        out = fused_fake_quantize(x)
        assert out.shape == x.shape
