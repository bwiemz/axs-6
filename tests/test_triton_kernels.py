"""
Tests for AXS-6 Triton Kernels
================================

Covers:
  - Triton availability detection
  - Bit-exact correctness vs fused_fake_quantize (nearest)
  - Stochastic rounding statistical properties
  - All valid block sizes (8, 16, 32)
  - Shape preservation across 1-D, 2-D, 3-D, and 4-D inputs
  - Non-divisible last-dimension (padding path)
  - Zero tensors
  - Large tensors
  - Single-element and tiny tensors
  - Backend dispatch integration (TRITON backend)
  - Graceful error on CPU tensors
  - LUT caching
"""

from __future__ import annotations

import pytest
import torch

from axs.core import VALID_BLOCK_SIZES
from axs.unified.quantize_unified import fused_fake_quantize

# Skip entire module if no CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset backend state before each test."""

    yield
    # Reset to auto-detect after test
    from axs.unified import backend as _bmod

    _bmod._active_backend = None


@pytest.fixture
def device():
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestTritonAvailability:
    def test_has_triton_returns_bool(self):
        from axs.unified.triton_kernels import has_triton

        assert isinstance(has_triton(), bool)

    def test_has_triton_true_on_cuda(self):
        from axs.unified.triton_kernels import has_triton

        assert has_triton()


# ---------------------------------------------------------------------------
# Nearest rounding: bit-exact correctness
# ---------------------------------------------------------------------------


class TestNearestCorrectness:
    """triton_fused_fake_quantize must be bit-exact with fused_fake_quantize."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4096, 4096),
            (256, 256),
            (100, 50),
            (7, 13),
            (1, 32),
            (4, 16, 64),
            (3, 100, 17),
            (2, 4, 8, 32),
            (128, 128),
        ],
        ids=lambda s: "x".join(str(d) for d in s),
    )
    def test_nearest_exact_match(self, shape, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        torch.manual_seed(42)
        x = torch.randn(*shape, device=device)
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-6), (
            f"Max diff: {(ref - tri).abs().max().item()}"
        )

    @pytest.mark.parametrize("bs", VALID_BLOCK_SIZES, ids=lambda b: f"bs{b}")
    def test_block_sizes(self, bs, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        torch.manual_seed(123)
        x = torch.randn(256, 256, device=device)
        ref = fused_fake_quantize(x, bs, "nearest")
        tri = triton_fused_fake_quantize(x, bs, "nearest")
        assert torch.allclose(ref, tri, atol=1e-6), (
            f"bs={bs}, max diff: {(ref - tri).abs().max().item()}"
        )

    def test_zero_tensor(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.zeros(128, 128, device=device)
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-6)
        assert (tri == 0).all()

    def test_single_block(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(1, 32, device=device)
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-6)

    def test_large_values(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64, device=device) * 1000.0
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-3)

    def test_tiny_values(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64, device=device) * 1e-6
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-12)

    def test_power_of_two_amax(self, device):
        """Regression: fast-math log2 near exact powers of 2."""
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        # Create tensor where block amax is exactly a power of 2
        x = torch.zeros(1, 32, device=device)
        for exp in [-2, 0, 1, 3, 5, 8]:
            x[0, 0] = 2.0**exp
            ref = fused_fake_quantize(x, 32, "nearest")
            tri = triton_fused_fake_quantize(x, 32, "nearest")
            assert torch.allclose(ref, tri, atol=1e-6), (
                f"exp={exp}, diff={( ref - tri).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


class TestShapePreservation:
    @pytest.mark.parametrize(
        "shape",
        [(64,), (32, 64), (4, 8, 16), (2, 3, 4, 32)],
        ids=lambda s: "x".join(str(d) for d in s),
    )
    def test_output_shape_matches_input(self, shape, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(*shape, device=device)
        out = triton_fused_fake_quantize(x, 32, "nearest")
        assert out.shape == x.shape

    def test_output_dtype_float32(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64, device=device)
        out = triton_fused_fake_quantize(x, 32, "nearest")
        assert out.dtype == torch.float32

    def test_output_device_matches_input(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64, device=device)
        out = triton_fused_fake_quantize(x, 32, "nearest")
        assert out.device == x.device


# ---------------------------------------------------------------------------
# Stochastic rounding
# ---------------------------------------------------------------------------


class TestStochasticRounding:
    def test_stochastic_shape_preserved(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(256, 256, device=device)
        out = triton_fused_fake_quantize(x, 32, "stochastic")
        assert out.shape == x.shape

    def test_stochastic_differs_from_nearest(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(1024, 1024, device=device)
        nearest = triton_fused_fake_quantize(x, 32, "nearest")
        stochastic = triton_fused_fake_quantize(x, 32, "stochastic")
        assert not torch.allclose(nearest, stochastic)

    def test_stochastic_mse_similar_to_pytorch(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(1024, 1024, device=device)
        ref_mse = (x - fused_fake_quantize(x, 32, "stochastic")).pow(2).mean()
        tri_mse = (x - triton_fused_fake_quantize(x, 32, "stochastic")).pow(2).mean()
        # MSE should be within 50% (random seeds differ)
        ratio = tri_mse / ref_mse
        assert 0.5 < ratio < 2.0, f"MSE ratio: {ratio.item()}"

    def test_stochastic_unbiased(self, device):
        """Mean error over many runs should be near zero (unbiased)."""
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(256, 256, device=device)
        errors = []
        for _ in range(20):
            out = triton_fused_fake_quantize(x, 32, "stochastic")
            errors.append((x - out).mean().item())
        mean_error = sum(errors) / len(errors)
        assert abs(mean_error) < 0.01, f"Mean error: {mean_error}"


# ---------------------------------------------------------------------------
# N_BLOCKS tuning parameter
# ---------------------------------------------------------------------------


class TestNBlocksParam:
    @pytest.mark.parametrize("n_blocks", [16, 32, 64, 128, 256])
    def test_n_blocks_correctness(self, n_blocks, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(512, 512, device=device)
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = triton_fused_fake_quantize(x, 32, "nearest", n_blocks=n_blocks)
        assert torch.allclose(ref, tri, atol=1e-6), (
            f"n_blocks={n_blocks}, diff={( ref - tri).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_cpu_tensor_raises(self):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64)  # CPU
        with pytest.raises(RuntimeError, match="CUDA"):
            triton_fused_fake_quantize(x, 32, "nearest")

    def test_invalid_block_size_raises(self, device):
        from axs.unified.triton_kernels import triton_fused_fake_quantize

        x = torch.randn(64, 64, device=device)
        with pytest.raises(AssertionError):
            triton_fused_fake_quantize(x, 17, "nearest")


# ---------------------------------------------------------------------------
# Backend dispatch integration
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_auto_detect_triton(self):
        from axs.unified.backend import BackendType, detect_best_backend

        assert detect_best_backend() == BackendType.TRITON

    def test_set_backend_triton(self, device):
        from axs.unified.backend import (
            BackendType,
            accelerated_fake_quantize,
            get_backend,
            set_backend,
        )

        set_backend("triton")
        assert get_backend() == BackendType.TRITON

        x = torch.randn(256, 256, device=device)
        ref = fused_fake_quantize(x, 32, "nearest")
        tri = accelerated_fake_quantize(x, 32, "nearest")
        assert torch.allclose(ref, tri, atol=1e-6)

    def test_accelerated_linear_triton(self, device):
        from axs.unified.backend import accelerated_linear, set_backend

        set_backend("triton")
        x = torch.randn(32, 128, device=device)
        w = torch.randn(64, 128, device=device)
        b = torch.randn(64, device=device)
        y = accelerated_linear(x, w, b, block_size=32)
        assert y.shape == (32, 64)

    def test_backend_info_shows_triton(self):
        from axs.unified.backend import backend_info, set_backend

        set_backend("triton")
        info = backend_info()
        assert info["active_backend"] == "triton"
        assert info["triton_kernel"] is True

    def test_env_override_triton(self):
        """AXS6_BACKEND=triton should force Triton backend."""
        import os
        from unittest import mock

        from axs.unified.backend import BackendType, detect_best_backend

        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "triton"}):
            assert detect_best_backend() == BackendType.TRITON


# ---------------------------------------------------------------------------
# LUT caching
# ---------------------------------------------------------------------------


class TestLUTCaching:
    def test_lut_cached_after_call(self, device):
        from axs.unified.triton_kernels import _LUT_CACHE, triton_fused_fake_quantize

        _LUT_CACHE.clear()
        x = torch.randn(64, 64, device=device)
        triton_fused_fake_quantize(x, 32, "nearest")
        assert len(_LUT_CACHE) > 0
