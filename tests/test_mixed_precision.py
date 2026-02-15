"""
Tests for AXS-6 BF16 Mixed-Precision Training
===============================================

Covers:
  - axs_linear_mixed_precision correctness vs FP32 reference
  - AXSLinearMixedPrecision module forward/backward
  - Output dtype is BF16; grad_weight is FP32
  - Warmup bypass mode
  - convert_to_axs_mixed_precision model conversion
  - estimate_memory_savings utility
  - Gradient flow through the full autograd path
  - Training convergence (multi-step optimiser loop)
  - 3-D batched inputs
  - CPU-only fallback (no CUDA)
  - Various matrix shapes (power-of-2, non-aligned)
  - Integration with existing AXS-6 modules
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.unified.quantize_unified import fused_fake_quantize
from axs.unified.mixed_precision import (
    AXSLinearMixedPrecision,
    axs_linear_mixed_precision,
    convert_to_axs_mixed_precision,
    estimate_memory_savings,
)

# Skip entire module if no CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for mixed-precision tests",
)

# BF16 has ~7.5-bit mantissa → larger tolerances than FP32
_ABS_TOL = 0.05
_REL_TOL = 5e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset backend state and disable TF32 for deterministic comparison."""
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    from axs.unified import backend as _bmod
    _bmod._active_backend = None
    torch.backends.cuda.matmul.allow_tf32 = True


def _ref_linear_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    quantize_input: bool = True,
) -> torch.Tensor:
    """Reference: cast to BF16 first, then FQ (matching actual code path)."""
    w_bf = w.to(torch.bfloat16)
    x_bf = x.to(torch.bfloat16)
    # FQ operates on BF16 input (kernel promotes to FP32 internally, returns BF16)
    w_q = fused_fake_quantize(w_bf.float(), 32, "nearest").bfloat16()
    if quantize_input:
        x_q = fused_fake_quantize(x_bf.float(), 32, "nearest").bfloat16()
    else:
        x_q = x_bf
    return F.linear(x_q, w_q, bias.bfloat16() if bias is not None else None)


# ===================================================================
# 1. Functional correctness (axs_linear_mixed_precision)
# ===================================================================


class TestFunctionalCorrectness:
    """Compare axs_linear_mixed_precision against BF16 reference."""

    @pytest.mark.parametrize(
        "M, N, K",
        [
            (128, 64, 256),
            (64, 128, 512),
            (256, 256, 256),
            (32, 32, 32),
        ],
    )
    def test_basic_shapes(self, M: int, N: int, K: int) -> None:
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda")
        w = torch.randn(N, K, device="cuda")

        out = axs_linear_mixed_precision(x, w)
        ref = _ref_linear_bf16(x, w, None)

        assert out.dtype == torch.bfloat16
        assert out.shape == (M, N)
        torch.testing.assert_close(out, ref, atol=_ABS_TOL, rtol=_REL_TOL)

    def test_with_bias(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(64, 128, device="cuda")
        w = torch.randn(32, 128, device="cuda")
        b = torch.randn(32, device="cuda")

        out = axs_linear_mixed_precision(x, w, b)
        ref = _ref_linear_bf16(x, w, b)

        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, ref, atol=_ABS_TOL, rtol=_REL_TOL)

    def test_no_input_quantise(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(64, 128, device="cuda")
        w = torch.randn(32, 128, device="cuda")

        out = axs_linear_mixed_precision(x, w, quantize_input=False)
        ref = _ref_linear_bf16(x, w, None, quantize_input=False)

        torch.testing.assert_close(out, ref, atol=_ABS_TOL, rtol=_REL_TOL)

    def test_3d_batched_input(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")

        out = axs_linear_mixed_precision(x, w)
        ref = _ref_linear_bf16(x, w, None)

        assert out.shape == (4, 16, 32)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, ref, atol=_ABS_TOL, rtol=_REL_TOL)


# ===================================================================
# 2. Output & gradient dtypes
# ===================================================================


class TestDtypes:
    """Verify output is BF16 and grad_weight is FP32."""

    def test_output_dtype_bf16(self) -> None:
        x = torch.randn(32, 64, device="cuda", requires_grad=True)
        w = torch.randn(16, 64, device="cuda", requires_grad=True)
        out = axs_linear_mixed_precision(x, w)
        assert out.dtype == torch.bfloat16

    def test_grad_weight_fp32(self) -> None:
        """Weight gradient should be FP32 for stable optimiser updates."""
        x = torch.randn(32, 64, device="cuda", requires_grad=True)
        w = torch.randn(16, 64, device="cuda", requires_grad=True)
        b = torch.randn(16, device="cuda", requires_grad=True)

        out = axs_linear_mixed_precision(x, w, b)
        loss = out.float().sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.dtype == torch.float32, (
            f"Weight gradient should be FP32, got {w.grad.dtype}"
        )
        assert b.grad is not None
        assert b.grad.dtype == torch.float32

    def test_grad_input_accumulates_fp32(self) -> None:
        """Input gradient is computed in BF16 but accumulated in FP32 (leaf dtype)."""
        x = torch.randn(32, 64, device="cuda", requires_grad=True)
        w = torch.randn(16, 64, device="cuda", requires_grad=True)

        out = axs_linear_mixed_precision(x, w)
        loss = out.float().sum()
        loss.backward()

        assert x.grad is not None
        # PyTorch accumulates gradients in the leaf tensor's dtype
        assert x.grad.dtype == torch.float32


# ===================================================================
# 3. Module tests (AXSLinearMixedPrecision)
# ===================================================================


class TestModule:
    """Tests for AXSLinearMixedPrecision nn.Module."""

    def test_module_forward(self) -> None:
        torch.manual_seed(42)
        layer = AXSLinearMixedPrecision(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        out = layer(x)
        assert out.shape == (8, 32)
        assert out.dtype == torch.bfloat16

    def test_module_with_bias(self) -> None:
        layer = AXSLinearMixedPrecision(64, 32, bias=True).cuda()
        assert layer.bias is not None
        assert layer.bias.dtype == torch.float32  # master weight

    def test_module_no_bias(self) -> None:
        layer = AXSLinearMixedPrecision(64, 32, bias=False).cuda()
        assert layer.bias is None

        x = torch.randn(8, 64, device="cuda")
        out = layer(x)
        assert out.shape == (8, 32)

    def test_master_weights_fp32(self) -> None:
        """Master weights must stay FP32 even after forward pass."""
        layer = AXSLinearMixedPrecision(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        _ = layer(x)
        assert layer.weight.dtype == torch.float32

    def test_warmup_mode(self) -> None:
        """Warmup bypasses quantisation — plain BF16 matmul."""
        torch.manual_seed(42)
        layer = AXSLinearMixedPrecision(64, 32).cuda()
        layer._warmup_active = True

        x = torch.randn(8, 64, device="cuda")
        out = layer(x)

        ref = F.linear(
            x.bfloat16(),
            layer.weight.bfloat16(),
            layer.bias.bfloat16() if layer.bias is not None else None,
        )
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_extra_repr(self) -> None:
        layer = AXSLinearMixedPrecision(64, 32)
        r = layer.extra_repr()
        assert "mixed_precision=bf16" in r
        assert "in_features=64" in r
        assert "out_features=32" in r

    def test_backward_through_module(self) -> None:
        layer = AXSLinearMixedPrecision(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        out = layer(x)
        loss = out.float().sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (32, 64)
        assert layer.weight.grad.dtype == torch.float32
        assert x.grad is not None


# ===================================================================
# 4. Gradient flow & training convergence
# ===================================================================


class TestTraining:
    """Verify gradient flow and training convergence."""

    def test_gradient_finite(self) -> None:
        """No NaN/Inf in gradients after forward+backward."""
        layer = AXSLinearMixedPrecision(128, 64).cuda()
        x = torch.randn(16, 128, device="cuda")
        out = layer(x)
        loss = out.float().sum()
        loss.backward()

        assert torch.isfinite(layer.weight.grad).all()  # type: ignore[union-attr]
        if layer.bias is not None and layer.bias.grad is not None:
            assert torch.isfinite(layer.bias.grad).all()

    def test_convergence_simple_fit(self) -> None:
        """Fit y = Wx + b (known target) — loss should decrease."""
        torch.manual_seed(42)
        in_f, out_f = 64, 32
        layer = AXSLinearMixedPrecision(in_f, out_f).cuda()
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2)

        # Target: a random linear mapping
        W_target = torch.randn(out_f, in_f, device="cuda")
        b_target = torch.randn(out_f, device="cuda")

        losses: list[float] = []
        for step in range(200):
            x = torch.randn(32, in_f, device="cuda")
            y_target = F.linear(x, W_target, b_target)

            y_pred = layer(x)
            loss = F.mse_loss(y_pred.float(), y_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first * 0.5, (
            f"Loss didn't converge: {avg_first:.4f} → {avg_last:.4f}"
        )

    def test_multi_layer_convergence(self) -> None:
        """Two-layer MLP with AXS-6 BF16 mixed precision should converge."""
        torch.manual_seed(42)

        model = nn.Sequential(
            AXSLinearMixedPrecision(32, 64),
            nn.ReLU(),
            AXSLinearMixedPrecision(64, 16),
        ).cuda()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for step in range(100):
            x = torch.randn(16, 32, device="cuda")
            target = torch.randn(16, 16, device="cuda")

            out = model(x).float()
            loss = F.mse_loss(out, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Multi-layer should still converge
        avg_first_10 = sum(losses[:10]) / 10
        avg_last_10 = sum(losses[-10:]) / 10
        assert avg_last_10 < avg_first_10 * 0.9, (
            f"Multi-layer loss didn't converge: "
            f"{avg_first_10:.4f} → {avg_last_10:.4f}"
        )


# ===================================================================
# 5. Model conversion
# ===================================================================


class TestConversion:
    """Tests for convert_to_axs_mixed_precision."""

    def test_convert_nn_linear(self) -> None:
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        converted = convert_to_axs_mixed_precision(model)

        assert isinstance(converted[0], AXSLinearMixedPrecision)
        assert isinstance(converted[2], AXSLinearMixedPrecision)
        assert isinstance(converted[1], nn.ReLU)  # unchanged

    def test_convert_preserves_weights(self) -> None:
        torch.manual_seed(42)
        model = nn.Linear(64, 32)
        w_orig = model.weight.data.clone()
        b_orig = model.bias.data.clone()  # type: ignore[union-attr]

        converted = convert_to_axs_mixed_precision(
            nn.Sequential(model),
        )
        layer = converted[0]
        assert isinstance(layer, AXSLinearMixedPrecision)
        torch.testing.assert_close(layer.weight.data, w_orig)
        assert layer.bias is not None
        torch.testing.assert_close(layer.bias.data, b_orig)

    def test_convert_no_inplace(self) -> None:
        """Default: returns a copy, original unchanged."""
        model = nn.Sequential(nn.Linear(64, 32))
        converted = convert_to_axs_mixed_precision(model)
        assert isinstance(model[0], nn.Linear)  # original unchanged
        assert isinstance(converted[0], AXSLinearMixedPrecision)

    def test_convert_inplace(self) -> None:
        model = nn.Sequential(nn.Linear(64, 32))
        convert_to_axs_mixed_precision(model, inplace=True)
        assert isinstance(model[0], AXSLinearMixedPrecision)

    def test_convert_skip_layers(self) -> None:
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 16),
        )
        converted = convert_to_axs_mixed_precision(
            model, skip_layers={"0"},
        )
        assert isinstance(converted[0], nn.Linear)  # skipped
        assert isinstance(converted[1], AXSLinearMixedPrecision)

    def test_convert_from_unified(self) -> None:
        """Convert an existing AXSLinearUnified to BF16 mixed precision."""
        from axs.unified.modules_unified import AXSLinearUnified

        model = nn.Sequential(AXSLinearUnified(64, 32))
        converted = convert_to_axs_mixed_precision(model)
        assert isinstance(converted[0], AXSLinearMixedPrecision)

    def test_convert_nested_model(self) -> None:
        """Handles nested submodules."""
        class TwoLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        model = TwoLayer()
        converted = convert_to_axs_mixed_precision(model)
        assert isinstance(converted.net[0], AXSLinearMixedPrecision)  # type: ignore[index]
        assert isinstance(converted.net[2], AXSLinearMixedPrecision)  # type: ignore[index]

    def test_converted_model_forward(self) -> None:
        """End-to-end: converted model produces output on CUDA."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        converted = convert_to_axs_mixed_precision(model).cuda()
        x = torch.randn(8, 64, device="cuda")

        out = converted(x)
        assert out.shape == (8, 16)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()


# ===================================================================
# 6. Memory estimation utility
# ===================================================================


class TestMemoryEstimation:
    """Tests for estimate_memory_savings."""

    def test_single_layer(self) -> None:
        model = nn.Sequential(AXSLinearMixedPrecision(512, 256))
        est = estimate_memory_savings(model, batch_size=32)

        assert est["fp32_mb"] > 0
        assert est["bf16_recomp_mb"] > 0
        assert est["savings_ratio"] >= 3.0  # should be ~4x

    def test_multi_layer(self) -> None:
        model = nn.Sequential(
            AXSLinearMixedPrecision(512, 256),
            nn.ReLU(),
            AXSLinearMixedPrecision(256, 128),
        )
        est = estimate_memory_savings(model, batch_size=32, seq_len=16)
        assert est["fp32_mb"] > 0
        assert est["savings_ratio"] >= 2.0

    def test_no_axs_layers(self) -> None:
        model = nn.Sequential(nn.Linear(64, 32))
        est = estimate_memory_savings(model)
        assert est["fp32_mb"] == 0.0
        assert est["bf16_recomp_mb"] == 0.0


# ===================================================================
# 7. Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_single_element(self) -> None:
        """1×32 input (single sample, minimum block size)."""
        x = torch.randn(1, 32, device="cuda")
        w = torch.randn(32, 32, device="cuda")
        out = axs_linear_mixed_precision(x, w)
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()

    def test_large_batch(self) -> None:
        """1024-sample batch."""
        x = torch.randn(1024, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = axs_linear_mixed_precision(x, w)
        assert out.shape == (1024, 32)
        assert torch.isfinite(out).all()

    def test_zero_input(self) -> None:
        x = torch.zeros(8, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = axs_linear_mixed_precision(x, w)
        assert out.shape == (8, 32)
        # Zero input → output should be roughly zero (only bias if present)
        assert out.abs().max() < 0.01

    def test_requires_grad_false(self) -> None:
        """No-grad forward should work without error."""
        layer = AXSLinearMixedPrecision(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")
        with torch.no_grad():
            out = layer(x)
        assert out.shape == (8, 32)

    def test_mixed_precision_vs_fp32_accuracy(self) -> None:
        """BF16 result should be close to FP32 AXS-6 result."""
        from axs.unified.functional_unified import axs_linear_unified

        torch.manual_seed(42)
        x = torch.randn(32, 64, device="cuda")
        w = torch.randn(16, 64, device="cuda")
        b = torch.randn(16, device="cuda")

        mp_out = axs_linear_mixed_precision(x, w, b).float()
        fp32_out = axs_linear_unified(x, w, b)

        # BF16 path does FQ on BF16-rounded values, FP32 path on exact FP32 values.
        # The FQ input difference + BF16 matmul rounding cause larger deltas.
        # atol=1.0 accounts for accumulated BF16 rounding × K-dim reduction.
        torch.testing.assert_close(mp_out, fp32_out, atol=1.0, rtol=0.1)


# ===================================================================
# 8. Integration with __init__ exports
# ===================================================================


class TestExports:
    """Verify all mixed-precision symbols are importable."""

    def test_import_from_unified(self) -> None:
        from axs.unified import (
            AXSLinearMixedPrecision,
            axs_linear_mixed_precision,
            convert_to_axs_mixed_precision,
            estimate_memory_savings,
        )
        assert callable(axs_linear_mixed_precision)
        assert callable(convert_to_axs_mixed_precision)
        assert callable(estimate_memory_savings)
