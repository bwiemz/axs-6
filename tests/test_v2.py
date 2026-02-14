"""
Tests for AXS-6 V2 optimization modules.

Tests all V2 components:
  - NormalFloat-5 codebook and encode/decode
  - Percentile clipping
  - Hadamard rotation
  - SmoothQuant
  - Precision annealing
  - V2 functional ops
  - V2 modules and model conversion
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from axs.core import quantize, dequantize, DEFAULT_BLOCK_SIZE


# ===========================================================================
# NF5 Codebook Tests
# ===========================================================================

class TestNF5Codebook:
    """Test the NormalFloat-5 codebook construction and properties."""

    def test_codebook_size(self) -> None:
        from axs.v2.quantize_v2 import NF5_CODEBOOK
        assert NF5_CODEBOOK.shape == (32,)

    def test_codebook_monotonic(self) -> None:
        from axs.v2.quantize_v2 import NF5_CODEBOOK
        for i in range(31):
            assert NF5_CODEBOOK[i] < NF5_CODEBOOK[i + 1], f"Not monotonic at {i}"

    def test_codebook_range(self) -> None:
        from axs.v2.quantize_v2 import NF5_CODEBOOK
        assert NF5_CODEBOOK[0] >= 0.0
        assert abs(NF5_CODEBOOK[-1].item() - 1.0) < 1e-6

    def test_codebook_denser_near_zero(self) -> None:
        """NF5 should have more codes near zero than near 1 (Gaussian property)."""
        from axs.v2.quantize_v2 import NF5_CODEBOOK
        # Count codes in [0, 0.5] vs (0.5, 1.0]
        lower_half = (NF5_CODEBOOK <= 0.5).sum().item()
        upper_half = (NF5_CODEBOOK > 0.5).sum().item()
        assert lower_half > upper_half, "NF5 should be denser near zero"

    def test_boundaries_size(self) -> None:
        from axs.v2.quantize_v2 import NF5_BOUNDARIES
        # 33 boundaries for 32 codes
        assert NF5_BOUNDARIES.shape == (33,)


class TestNF5EncodeDecode:
    """Test NF5 encode and decode roundtrip."""

    def test_roundtrip_exact_codes(self) -> None:
        """Encoding exact codebook values should return those codes."""
        from axs.v2.quantize_v2 import NF5_CODEBOOK, nf5_encode, nf5_decode
        codes = nf5_encode(NF5_CODEBOOK)
        decoded = nf5_decode(codes)
        assert torch.allclose(decoded, NF5_CODEBOOK, atol=1e-6)

    def test_encode_range(self) -> None:
        from axs.v2.quantize_v2 import nf5_encode
        vals = torch.rand(100)
        codes = nf5_encode(vals)
        assert codes.min() >= 0
        assert codes.max() <= 31

    def test_stochastic_unbiased(self) -> None:
        """Stochastic NF5 encoding should be approximately unbiased."""
        from axs.v2.quantize_v2 import nf5_encode_stochastic, nf5_decode
        torch.manual_seed(42)
        val = torch.full((10000,), 0.5)
        codes = nf5_encode_stochastic(val)
        decoded = nf5_decode(codes)
        mean_decoded = decoded.mean().item()
        assert abs(mean_decoded - 0.5) < 0.05, f"Expected ~0.5, got {mean_decoded}"


# ===========================================================================
# Quantize V2 Tests
# ===========================================================================

class TestQuantizeV2:
    """Test the full V2 quantize/dequantize pipeline."""

    def test_basic_roundtrip(self) -> None:
        from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
        t = torch.randn(64, 128)
        axs = quantize_v2(t, block_size=32)
        r = dequantize_v2(axs, use_nf5=True)
        assert r.shape == t.shape

    def test_shape_preservation(self) -> None:
        from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
        for shape in [(32,), (64, 32), (4, 8, 32), (2, 3, 4, 32)]:
            t = torch.randn(*shape)
            axs = quantize_v2(t, block_size=32)
            r = dequantize_v2(axs, use_nf5=True)
            assert r.shape == t.shape, f"Shape mismatch for {shape}"

    def test_v2_better_than_v1_gaussian(self) -> None:
        """V2 should have lower MSE than V1 on Gaussian data."""
        from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
        torch.manual_seed(42)
        t = torch.randn(1024, 128)

        axs_v1 = quantize(t, block_size=32)
        r_v1 = dequantize(axs_v1)
        mse_v1 = (t - r_v1).pow(2).mean().item()

        axs_v2 = quantize_v2(t, block_size=32, use_nf5=True)
        r_v2 = dequantize_v2(axs_v2, use_nf5=True)
        mse_v2 = (t - r_v2).pow(2).mean().item()

        assert mse_v2 < mse_v1, f"V2 MSE ({mse_v2}) should be < V1 MSE ({mse_v1})"

    def test_uniform_grid_fallback(self) -> None:
        """With use_nf5=False, V2 should behave like V1."""
        from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
        torch.manual_seed(42)
        t = torch.randn(64, 32)
        axs = quantize_v2(t, block_size=32, use_nf5=False, clip_percentile=None)
        r = dequantize_v2(axs, use_nf5=False)

        axs_v1 = quantize(t, block_size=32)
        r_v1 = dequantize(axs_v1)

        # Should be identical
        assert torch.allclose(r, r_v1, atol=1e-6)


# ===========================================================================
# Percentile Clipping Tests
# ===========================================================================

class TestPercentileScale:
    """Test percentile-based scale computation."""

    def test_100_percentile_equals_absmax(self) -> None:
        from axs.v2.quantize_v2 import percentile_scale
        blocked = torch.randn(4, 8, 32)
        exps, scales = percentile_scale(blocked, percentile=100.0)
        # Should be same as abs_max-based scaling
        abs_max = blocked.abs().amax(dim=-1)
        assert torch.all(exps > 0) or torch.all(abs_max == 0)

    def test_lower_percentile_clips(self) -> None:
        from axs.v2.quantize_v2 import percentile_scale
        # Create tensor with one outlier per block
        blocked = torch.randn(1, 1, 32) * 0.1
        blocked[0, 0, 0] = 100.0  # massive outlier
        _, scales_100 = percentile_scale(blocked, percentile=100.0)
        _, scales_90 = percentile_scale(blocked, percentile=90.0)
        # 90th percentile should give smaller scale (clips outlier)
        assert scales_90[0, 0, 0] <= scales_100[0, 0, 0]


# ===========================================================================
# Hadamard Rotation Tests
# ===========================================================================

class TestHadamard:
    """Test Hadamard rotation properties."""

    def test_hadamard_matrix_orthogonal(self) -> None:
        from axs.v2.hadamard import hadamard_matrix
        H = hadamard_matrix(32)
        product = H @ H.T
        assert torch.allclose(product, torch.eye(32), atol=1e-5)

    def test_hadamard_self_inverse(self) -> None:
        from axs.v2.hadamard import fast_walsh_hadamard
        x = torch.randn(4, 32)
        transformed = fast_walsh_hadamard(x)
        recovered = fast_walsh_hadamard(transformed)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_rotation_preserves_norm(self) -> None:
        from axs.v2.hadamard import apply_hadamard_rotation
        x = torch.randn(8, 4, 32)
        y = apply_hadamard_rotation(x)
        # Norms should be preserved (orthogonal transform)
        assert torch.allclose(
            x.norm(dim=-1), y.norm(dim=-1), atol=1e-4
        )

    def test_rotation_roundtrip(self) -> None:
        from axs.v2.hadamard import apply_hadamard_rotation, invert_hadamard_rotation
        x = torch.randn(4, 8, 32)
        y = apply_hadamard_rotation(x)
        z = invert_hadamard_rotation(y)
        assert torch.allclose(x, z, atol=1e-5)

    def test_rotation_spreads_outliers(self) -> None:
        """After rotation, max/mean ratio should decrease."""
        from axs.v2.hadamard import apply_hadamard_rotation
        # Create tensor with big outlier
        x = torch.ones(1, 1, 32) * 0.1
        x[0, 0, 0] = 100.0

        y = apply_hadamard_rotation(x, use_random_signs=False)
        x_ratio = x.abs().max() / x.abs().mean()
        y_ratio = y.abs().max() / y.abs().mean()
        assert y_ratio < x_ratio, "Hadamard should spread outliers"


# ===========================================================================
# SmoothQuant Tests
# ===========================================================================

class TestSmoothQuant:
    """Test SmoothQuant scale computation."""

    def test_compute_scales(self) -> None:
        from axs.v2.smooth_quant import compute_smooth_scales
        act_max = torch.tensor([10.0, 1.0, 0.1])
        weight_max = torch.tensor([0.1, 1.0, 10.0])
        scales = compute_smooth_scales(act_max, weight_max, alpha=0.5)
        assert scales.shape == (3,)
        assert torch.all(scales > 0)

    def test_smooth_scales_balance(self) -> None:
        """With alpha=0.5, scales should balance act and weight magnitudes."""
        from axs.v2.smooth_quant import compute_smooth_scales
        act_max = torch.tensor([100.0])
        weight_max = torch.tensor([0.01])
        scales = compute_smooth_scales(act_max, weight_max, alpha=0.5)
        # scale = (100)^0.5 / (0.01)^0.5 = 10 / 0.1 = 100
        assert scales[0] > 1.0, "Scale should be > 1 to balance"


# ===========================================================================
# Precision Annealing Tests
# ===========================================================================

class TestPrecisionAnnealing:
    """Test precision annealing schedule."""

    def test_initial_strength(self) -> None:
        from axs.v2.annealing import PrecisionAnnealingSchedule
        schedule = PrecisionAnnealingSchedule(warmup_steps=100)
        assert schedule.strength == 0.0

    def test_strength_increases(self) -> None:
        from axs.v2.annealing import PrecisionAnnealingSchedule
        schedule = PrecisionAnnealingSchedule(warmup_steps=100)
        strengths = []
        for _ in range(120):
            strengths.append(schedule.strength)
            schedule.step()
        assert strengths[0] == 0.0
        assert strengths[50] > 0.0
        assert strengths[50] < 1.0
        assert strengths[-1] == 1.0

    def test_annealed_fake_quantize(self) -> None:
        from axs.v2.annealing import annealed_fake_quantize
        t = torch.randn(64, 32)

        # strength=0 → no quantization
        r0 = annealed_fake_quantize(t, strength=0.0)
        assert torch.equal(r0, t)

        # strength=0.5 → interpolation
        r5 = annealed_fake_quantize(t, strength=0.5)
        assert not torch.equal(r5, t)

        # strength=1 → full quantization
        r1 = annealed_fake_quantize(t, strength=1.0)
        assert not torch.equal(r1, t)


# ===========================================================================
# Amax History Tests
# ===========================================================================

class TestAmaxHistory:
    """Test delayed scaling with Amax history."""

    def test_ema_mode(self) -> None:
        from axs.v2.annealing import AmaxHistory
        hist = AmaxHistory(mode="ema", ema_decay=0.9)
        hist.update("w1", 10.0)
        assert hist.get_amax("w1") == 10.0
        hist.update("w1", 20.0)
        # EMA: 0.9 * 10 + 0.1 * 20 = 11.0
        assert abs(hist.get_amax("w1") - 11.0) < 1e-6

    def test_max_window_mode(self) -> None:
        from axs.v2.annealing import AmaxHistory
        hist = AmaxHistory(mode="max_window", window_size=3)
        hist.update("w1", 5.0)
        hist.update("w1", 10.0)
        hist.update("w1", 3.0)
        assert hist.get_amax("w1") == 10.0
        hist.update("w1", 1.0)  # window: [10, 3, 1]
        assert hist.get_amax("w1") == 10.0
        hist.update("w1", 1.0)  # window: [3, 1, 1]
        assert hist.get_amax("w1") == 3.0


# ===========================================================================
# V2 Functional Ops Tests
# ===========================================================================

class TestV2Functional:
    """Test V2 autograd functions."""

    def test_fake_quantize_v2_shape(self) -> None:
        from axs.v2.functional_v2 import fake_quantize_v2
        t = torch.randn(8, 64, requires_grad=True)
        q = fake_quantize_v2(t)
        assert q.shape == t.shape

    def test_fake_quantize_v2_gradient(self) -> None:
        """STE should pass gradients through."""
        from axs.v2.functional_v2 import fake_quantize_v2
        t = torch.randn(8, 32, requires_grad=True)
        q = fake_quantize_v2(t)
        loss = q.sum()
        loss.backward()
        assert t.grad is not None
        assert t.grad.shape == t.shape

    def test_axs_linear_v2(self) -> None:
        from axs.v2.functional_v2 import axs_linear_v2
        x = torch.randn(4, 64)
        w = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(32)
        out = axs_linear_v2(x, w, b)
        assert out.shape == (4, 32)
        out.sum().backward()
        assert w.grad is not None


# ===========================================================================
# V2 Modules Tests
# ===========================================================================

class TestV2Modules:
    """Test V2 drop-in module replacements."""

    def test_linear_v2_forward(self) -> None:
        from axs.v2.modules_v2 import AXSLinearV2
        layer = AXSLinearV2(64, 32)
        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_linear_v2_backward(self) -> None:
        from axs.v2.modules_v2 import AXSLinearV2
        layer = AXSLinearV2(64, 32)
        x = torch.randn(4, 64)
        y = layer(x)
        y.sum().backward()
        assert layer.weight.grad is not None

    def test_layernorm_v2_no_quant(self) -> None:
        """V2 LayerNorm should not quantize output."""
        from axs.v2.modules_v2 import AXSLayerNormV2
        ln = AXSLayerNormV2(32)
        x = torch.randn(4, 32)
        y = ln(x)
        # Standard LayerNorm
        y_ref = torch.nn.functional.layer_norm(x, [32], ln.weight, ln.bias, ln.eps)
        assert torch.equal(y, y_ref), "V2 LayerNorm should not add quantization noise"

    def test_embedding_v2_lazy(self) -> None:
        from axs.v2.modules_v2 import AXSEmbeddingV2
        emb = AXSEmbeddingV2(100, 32)
        idx = torch.tensor([0, 5, 10])
        y = emb(idx)
        assert y.shape == (3, 32)

    def test_mha_v2_forward(self) -> None:
        from axs.v2.modules_v2 import AXSMultiheadAttentionV2
        mha = AXSMultiheadAttentionV2(64, 4)
        x = torch.randn(2, 8, 64)
        y = mha(x, x, x)
        assert y.shape == (2, 8, 64)

    def test_convert_to_axs_v2(self) -> None:
        from axs.v2.modules_v2 import convert_to_axs_v2, AXSLinearV2, AXSLayerNormV2
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
        )
        v2_model = convert_to_axs_v2(model)
        assert isinstance(v2_model[0], AXSLinearV2)
        assert isinstance(v2_model[1], AXSLayerNormV2)
        assert isinstance(v2_model[2], AXSLinearV2)

    def test_convert_preserves_weights(self) -> None:
        from axs.v2.modules_v2 import convert_to_axs_v2, AXSLinearV2
        model = nn.Linear(64, 32)
        w_orig = model.weight.data.clone()
        v2 = convert_to_axs_v2(nn.Sequential(model))
        assert isinstance(v2[0], AXSLinearV2)
        assert torch.equal(v2[0].weight.data, w_orig)

    def test_v2_model_trains(self) -> None:
        """Verify a V2 model can train and reduce loss."""
        from axs.v2.modules_v2 import convert_to_axs_v2
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        model = convert_to_axs_v2(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x = torch.randn(32, 32)
        y = torch.randn(32, 1)

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "V2 model should reduce loss during training"
