"""
Tests for AXS-6 Fused FQ+Matmul Triton Kernel
================================================

Covers:
  - Correctness vs separate FQ + F.linear reference (nearest-mode)
  - Bias / no-bias paths
  - Input quantization on/off (QUANTIZE_X constexpr)
  - Various matrix shapes including non-power-of-2 and K not aligned to 32
  - 3-D batched inputs
  - Single-row (M=1) edge case
  - block_size validation (only 32 supported)
  - CPU tensor error handling
  - Backend dispatch integration via accelerated_linear
  - Output shape preservation
  - Gradient flow through the autograd path
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from axs.unified.quantize_unified import fused_fake_quantize

# Skip entire module if no CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton fused-linear tests",
)

# Tolerance for FP32 IEEE-precision matmul comparison
_ABS_TOL = 1e-3
_REL_TOL = 1e-5


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


def _ref_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    quantize_input: bool,
) -> torch.Tensor:
    """Reference: separate FQ + F.linear (TF32 already disabled by fixture)."""
    w_q = fused_fake_quantize(w, 32, "nearest")
    x_q = fused_fake_quantize(x, 32, "nearest") if quantize_input else x
    return F.linear(x_q, w_q, bias)


# ---------------------------------------------------------------------------
# Correctness — basic shapes
# ---------------------------------------------------------------------------


class TestFusedLinearCorrectness:
    """Compare fused kernel output against separate FQ + F.linear."""

    @pytest.mark.parametrize(
        "M, N, K",
        [
            (128, 64, 256),
            (64, 128, 512),
            (256, 256, 256),
            (32, 32, 32),
            (512, 512, 512),
            (1, 64, 128),     # single row
            (7, 13, 64),      # non-power-of-2 M, N
        ],
    )
    def test_with_bias(self, M: int, N: int, K: int) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda")
        w = torch.randn(N, K, device="cuda")
        b = torch.randn(N, device="cuda")

        ref = _ref_linear(x, w, b, quantize_input=True)
        fused = triton_fused_linear(x, w, b, 32, True)

        assert fused.shape == ref.shape
        assert torch.allclose(fused, ref, atol=_ABS_TOL, rtol=_REL_TOL), (
            f"max_diff={( ref - fused).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize(
        "M, N, K",
        [
            (128, 64, 256),
            (512, 512, 512),
            (1, 32, 64),
        ],
    )
    def test_without_bias(self, M: int, N: int, K: int) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda")
        w = torch.randn(N, K, device="cuda")

        ref = _ref_linear(x, w, None, quantize_input=True)
        fused = triton_fused_linear(x, w, None, 32, True)

        assert fused.shape == ref.shape
        assert torch.allclose(fused, ref, atol=_ABS_TOL, rtol=_REL_TOL), (
            f"max_diff={(ref - fused).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize(
        "M, N, K",
        [
            (128, 64, 256),
            (512, 512, 512),
        ],
    )
    def test_no_input_quantization(self, M: int, N: int, K: int) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda")
        w = torch.randn(N, K, device="cuda")
        b = torch.randn(N, device="cuda")

        ref = _ref_linear(x, w, b, quantize_input=False)
        fused = triton_fused_linear(x, w, b, 32, False)

        assert torch.allclose(fused, ref, atol=_ABS_TOL, rtol=_REL_TOL), (
            f"max_diff={(ref - fused).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Non-aligned K (not a multiple of 32)
# ---------------------------------------------------------------------------


class TestNonAlignedK:
    """Verify correctness when K is not a multiple of block_size=32."""

    @pytest.mark.parametrize("K", [33, 50, 100, 127, 255])
    def test_non_aligned_k(self, K: int) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        M, N = 64, 32
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda")
        w = torch.randn(N, K, device="cuda")

        ref = _ref_linear(x, w, None, quantize_input=True)
        fused = triton_fused_linear(x, w, None, 32, True)

        assert fused.shape == ref.shape
        assert torch.allclose(fused, ref, atol=_ABS_TOL, rtol=_REL_TOL), (
            f"K={K} max_diff={(ref - fused).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Batched / 3-D input
# ---------------------------------------------------------------------------


class TestBatchedInput:
    """Fused linear should handle (batch, seq, features) inputs."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 32, 256),        # (batch, seq, features)
            (2, 8, 16, 128),     # 4-D
            (1, 1, 64),          # minimal 3-D
        ],
    )
    def test_batched_shapes(self, shape: tuple[int, ...]) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        N = 64
        K = shape[-1]
        torch.manual_seed(42)
        x = torch.randn(*shape, device="cuda")
        w = torch.randn(N, K, device="cuda")
        b = torch.randn(N, device="cuda")

        ref = _ref_linear(x, w, b, quantize_input=True)
        fused = triton_fused_linear(x, w, b, 32, True)

        expected_shape = shape[:-1] + (N,)
        assert fused.shape == expected_shape
        assert torch.allclose(fused, ref, atol=_ABS_TOL, rtol=_REL_TOL), (
            f"shape={shape} max_diff={(ref - fused).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Validate proper errors for unsupported configurations."""

    def test_cpu_input_raises(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        x = torch.randn(4, 32)
        w = torch.randn(16, 32)
        with pytest.raises(RuntimeError, match="CUDA"):
            triton_fused_linear(x, w, None, 32, True)

    def test_block_size_not_32_raises(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        x = torch.randn(4, 32, device="cuda")
        w = torch.randn(16, 32, device="cuda")
        with pytest.raises(ValueError, match="block_size=32"):
            triton_fused_linear(x, w, None, 16, True)


# ---------------------------------------------------------------------------
# Backend dispatch integration
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    """Verify that accelerated_linear routes through the fused kernel."""

    def test_triton_backend_uses_fused(self) -> None:
        from axs.unified.backend import accelerated_linear, set_backend

        set_backend("triton")
        torch.manual_seed(42)
        x = torch.randn(64, 128, device="cuda")
        w = torch.randn(32, 128, device="cuda")
        b = torch.randn(32, device="cuda")

        result = accelerated_linear(x, w, b, block_size=32)
        ref = _ref_linear(x, w, b, quantize_input=True)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=_ABS_TOL, rtol=_REL_TOL)

    def test_triton_backend_non_32_falls_back(self) -> None:
        """block_size=16 should fall back to separate FQ path, not crash."""
        from axs.unified.backend import accelerated_linear, set_backend

        set_backend("triton")
        x = torch.randn(64, 128, device="cuda")
        w = torch.randn(32, 128, device="cuda")

        # Should not raise — falls back to separate FQ + F.linear
        result = accelerated_linear(x, w, None, block_size=16)
        assert result.shape == (64, 32)


# ---------------------------------------------------------------------------
# Autograd integration
# ---------------------------------------------------------------------------


class TestAutograd:
    """Verify gradients flow correctly through the fused linear path."""

    def test_gradient_flow(self) -> None:
        """Weights and input should receive gradients."""
        from axs.unified.backend import set_backend
        from axs.unified.functional_unified import axs_linear_unified

        set_backend("triton")
        x = torch.randn(16, 64, device="cuda", requires_grad=True)
        w = torch.randn(32, 64, device="cuda", requires_grad=True)
        b = torch.randn(32, device="cuda", requires_grad=True)

        out = axs_linear_unified(x, w, b, block_size=32)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "input gradient missing"
        assert w.grad is not None, "weight gradient missing"
        assert b.grad is not None, "bias gradient missing"
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape

    def test_gradient_magnitude_reasonable(self) -> None:
        """Grad norms should be finite and non-zero."""
        from axs.unified.backend import set_backend
        from axs.unified.functional_unified import axs_linear_unified

        set_backend("triton")
        torch.manual_seed(123)
        x = torch.randn(32, 128, device="cuda", requires_grad=True)
        w = torch.randn(64, 128, device="cuda", requires_grad=True)

        out = axs_linear_unified(x, w, None, block_size=32)
        out.sum().backward()

        for name, param in [("x", x), ("w", w)]:
            assert param.grad is not None
            norm = param.grad.norm().item()
            assert 0 < norm < 1e6, f"{name}.grad norm={norm} out of range"


# ---------------------------------------------------------------------------
# Output dtype and device
# ---------------------------------------------------------------------------


class TestOutputProperties:
    """Verify the output's dtype, device, and contiguity."""

    def test_output_float32(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        x = torch.randn(16, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = triton_fused_linear(x, w, None, 32, True)
        assert out.dtype == torch.float32

    def test_output_device(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        x = torch.randn(16, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = triton_fused_linear(x, w, None, 32, True)
        assert out.is_cuda

    def test_output_contiguous(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear

        x = torch.randn(16, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = triton_fused_linear(x, w, None, 32, True)
        assert out.is_contiguous()


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestExports:
    """Verify triton_fused_linear is reachable from the public API."""

    def test_import_from_triton_kernels(self) -> None:
        from axs.unified.triton_kernels import triton_fused_linear  # noqa: F401

    def test_import_from_unified(self) -> None:
        from axs.unified import triton_fused_linear  # noqa: F401
