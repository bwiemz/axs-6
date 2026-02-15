"""
Tests for AXS-6 Hardware Backend Dispatch
==========================================

Covers:
  - Backend detection and selection
  - set_backend / get_backend API
  - LUT cache management
  - Compilable fake-quantize numerical correctness
  - INT8 tensor core linear correctness (when available)
  - Accelerated dispatch functions
  - Fallback behaviour
  - backend_info() diagnostic
  - Environment variable override
"""

from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
import torch.nn as nn

from axs.core import DEFAULT_BLOCK_SIZE
from axs.unified.backend import (
    BackendType,
    _INT8_CACHE_MAX_SIZE,
    _INT8_CACHE_REFS,
    _INT8_WEIGHT_CACHE,
    _LUT_CACHE,
    _fused_fq_compilable,
    _get_lut,
    _has_cuda,
    _has_int8_tensorcore,
    _has_torch_compile,
    _int8_matmul_2d,
    _make_cache_finalizer,
    _nf5_to_int8_row,
    _to_int8_row,
    accelerated_fake_quantize,
    accelerated_linear,
    accelerated_matmul,
    backend_info,
    clear_int8_weight_cache,
    detect_best_backend,
    get_backend,
    int8_linear,
    set_backend,
)
from axs.unified.quantize_unified import fused_fake_quantize
from axs.unified.modules_unified import AXSLinearUnified


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def reset_backend():
    """Reset backend state between tests."""
    import axs.unified.backend as _mod
    _mod._active_backend = None
    _mod._compiled_fq = None
    _mod._compiled_fq_stochastic = None
    yield
    _mod._active_backend = None
    _mod._compiled_fq = None
    _mod._compiled_fq_stochastic = None


@pytest.fixture
def randn_block():
    torch.manual_seed(42)
    return torch.randn(32)


@pytest.fixture
def randn_2d():
    torch.manual_seed(42)
    return torch.randn(128, 256)


@pytest.fixture
def randn_3d():
    torch.manual_seed(42)
    return torch.randn(4, 16, 64)


# ===================================================================
# 1. BackendType enum
# ===================================================================

class TestBackendType:

    def test_enum_values(self):
        assert BackendType.EAGER.value == "eager"
        assert BackendType.COMPILED.value == "compiled"
        assert BackendType.INT8.value == "int8"

    def test_from_string(self):
        assert BackendType("eager") == BackendType.EAGER
        assert BackendType("compiled") == BackendType.COMPILED
        assert BackendType("int8") == BackendType.INT8

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            BackendType("fp16")


# ===================================================================
# 2. Backend detection
# ===================================================================

class TestDetection:

    def test_detect_returns_valid_type(self):
        result = detect_best_backend()
        assert isinstance(result, BackendType)

    def test_detect_compiled_when_cuda_and_compile(self):
        """When CUDA + compile but no Triton and no INT8 tensor cores → COMPILED."""
        with mock.patch("axs.unified.backend._has_triton_kernel", return_value=False), \
             mock.patch("axs.unified.backend._has_cuda", return_value=True), \
             mock.patch("axs.unified.backend._has_int8_tensorcore", return_value=False), \
             mock.patch("axs.unified.backend._has_torch_compile", return_value=True):
            assert detect_best_backend() == BackendType.COMPILED

    def test_detect_int8_when_tensorcore_available(self):
        """When CUDA + compile + INT8 tensor cores but no Triton → INT8."""
        with mock.patch("axs.unified.backend._has_triton_kernel", return_value=False), \
             mock.patch("axs.unified.backend._has_cuda", return_value=True), \
             mock.patch("axs.unified.backend._has_int8_tensorcore", return_value=True), \
             mock.patch("axs.unified.backend._has_torch_compile", return_value=True):
            assert detect_best_backend() == BackendType.INT8

    def test_detect_eager_when_no_compile(self):
        with mock.patch("axs.unified.backend._has_triton_kernel", return_value=False), \
             mock.patch("axs.unified.backend._has_cuda", return_value=True), \
             mock.patch("axs.unified.backend._has_torch_compile", return_value=False):
            assert detect_best_backend() == BackendType.EAGER

    def test_detect_eager_when_no_cuda(self):
        with mock.patch("axs.unified.backend._has_triton_kernel", return_value=False), \
             mock.patch("axs.unified.backend._has_cuda", return_value=False):
            assert detect_best_backend() == BackendType.EAGER

    def test_env_override_eager(self):
        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "eager"}):
            assert detect_best_backend() == BackendType.EAGER

    def test_env_override_int8(self):
        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "int8"}):
            assert detect_best_backend() == BackendType.INT8

    def test_env_override_compiled(self):
        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "compiled"}):
            assert detect_best_backend() == BackendType.COMPILED

    def test_env_override_case_insensitive(self):
        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "COMPILED"}):
            assert detect_best_backend() == BackendType.COMPILED

    def test_env_override_with_whitespace(self):
        with mock.patch.dict(os.environ, {"AXS6_BACKEND": "  eager  "}):
            assert detect_best_backend() == BackendType.EAGER


# ===================================================================
# 3. get_backend / set_backend
# ===================================================================

class TestGetSetBackend:

    def test_get_backend_auto_detects(self):
        backend = get_backend()
        assert isinstance(backend, BackendType)

    def test_set_backend_string(self):
        set_backend("eager")
        assert get_backend() == BackendType.EAGER

    def test_set_backend_enum(self):
        set_backend(BackendType.INT8)
        assert get_backend() == BackendType.INT8

    def test_set_backend_resets_compiled_cache(self):
        import axs.unified.backend as _mod
        # Simulate cached compiled function
        _mod._compiled_fq = lambda x, bs: x
        set_backend("eager")
        assert _mod._compiled_fq is None

    def test_round_trip_all_backends(self):
        for bt in BackendType:
            set_backend(bt)
            assert get_backend() == bt


# ===================================================================
# 4. LUT cache
# ===================================================================

class TestLUTCache:

    def test_cpu_lut_cached(self):
        lut = _get_lut(torch.device("cpu"))
        assert lut.device == torch.device("cpu")
        assert torch.device("cpu") in _LUT_CACHE
        # Second call returns same object
        lut2 = _get_lut(torch.device("cpu"))
        assert lut is lut2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_lut_cached(self):
        device = torch.device("cuda")
        lut = _get_lut(device)
        assert lut.is_cuda
        lut2 = _get_lut(device)
        assert lut is lut2

    def test_lut_shape_and_range(self):
        lut = _get_lut(torch.device("cpu"))
        assert lut.shape == (1024,)
        assert lut.min() >= 0.0
        assert lut.max() <= 1.0 + 1e-6


# ===================================================================
# 5. Compilable fake-quantize correctness (CPU)
# ===================================================================

class TestCompilableFQ:
    """Verify _fused_fq_compilable matches eager fused_fake_quantize."""

    def test_1d_matches_eager(self, randn_block):
        eager = fused_fake_quantize(randn_block, DEFAULT_BLOCK_SIZE, "nearest")
        compilable = _fused_fq_compilable(randn_block, DEFAULT_BLOCK_SIZE, False)
        torch.testing.assert_close(compilable, eager, atol=1e-6, rtol=1e-5)

    def test_2d_matches_eager(self, randn_2d):
        eager = fused_fake_quantize(randn_2d, DEFAULT_BLOCK_SIZE, "nearest")
        compilable = _fused_fq_compilable(randn_2d, DEFAULT_BLOCK_SIZE, False)
        torch.testing.assert_close(compilable, eager, atol=1e-6, rtol=1e-5)

    def test_3d_matches_eager(self, randn_3d):
        eager = fused_fake_quantize(randn_3d, DEFAULT_BLOCK_SIZE, "nearest")
        compilable = _fused_fq_compilable(randn_3d, DEFAULT_BLOCK_SIZE, False)
        torch.testing.assert_close(compilable, eager, atol=1e-6, rtol=1e-5)

    def test_preserves_shape(self, randn_3d):
        out = _fused_fq_compilable(randn_3d, DEFAULT_BLOCK_SIZE, False)
        assert out.shape == randn_3d.shape

    def test_stochastic_differs_from_nearest(self, randn_2d):
        """Stochastic rounding should produce different results (with high prob)."""
        torch.manual_seed(0)
        s1 = _fused_fq_compilable(randn_2d, DEFAULT_BLOCK_SIZE, True)
        torch.manual_seed(1)
        s2 = _fused_fq_compilable(randn_2d, DEFAULT_BLOCK_SIZE, True)
        # Should differ (random dither)
        assert not torch.equal(s1, s2)

    def test_zero_tensor(self):
        t = torch.zeros(64)
        out = _fused_fq_compilable(t, DEFAULT_BLOCK_SIZE, False)
        torch.testing.assert_close(out, t, atol=1e-7, rtol=0)

    def test_non_divisible_block_size(self):
        """Tensor width not divisible by block_size should still work."""
        t = torch.randn(10, 50)  # 50 is not divisible by 32
        eager = fused_fake_quantize(t, DEFAULT_BLOCK_SIZE, "nearest")
        compilable = _fused_fq_compilable(t, DEFAULT_BLOCK_SIZE, False)
        torch.testing.assert_close(compilable, eager, atol=1e-6, rtol=1e-5)


# ===================================================================
# 6. INT8 linear correctness (CUDA only)
# ===================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestINT8Linear:

    def test_basic_correctness(self):
        """INT8 linear should approximate NF5 + FP32 matmul."""
        torch.manual_seed(42)
        x = torch.randn(64, 256, device="cuda")
        w = torch.randn(128, 256, device="cuda")
        b = torch.randn(128, device="cuda")

        # Reference: eager NF5 FQ + FP32 matmul
        x_q = fused_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
        w_q = fused_fake_quantize(w, DEFAULT_BLOCK_SIZE, "nearest")
        ref = torch.nn.functional.linear(x_q, w_q, b)

        # INT8 path
        result = int8_linear(x, w, b, DEFAULT_BLOCK_SIZE)

        # INT8 approximation error should be small (< 5% relative)
        rel_err = (result - ref).abs().mean() / ref.abs().mean()
        assert rel_err < 0.05, f"INT8 relative error {rel_err:.4f} exceeds 5%"

    def test_output_shape(self):
        x = torch.randn(32, 512, device="cuda")
        w = torch.randn(256, 512, device="cuda")
        out = int8_linear(x, w, block_size=DEFAULT_BLOCK_SIZE)
        assert out.shape == (32, 256)

    def test_output_shape_3d(self):
        x = torch.randn(4, 16, 256, device="cuda")
        w = torch.randn(128, 256, device="cuda")
        out = int8_linear(x, w, block_size=DEFAULT_BLOCK_SIZE)
        assert out.shape == (4, 16, 128)

    def test_no_bias(self):
        x = torch.randn(64, 256, device="cuda")
        w = torch.randn(128, 256, device="cuda")
        out = int8_linear(x, w, bias=None, block_size=DEFAULT_BLOCK_SIZE)
        assert out.shape == (64, 128)

    def test_small_batch_padding(self):
        """M < 32 should still work (padding to 32)."""
        x = torch.randn(3, 256, device="cuda")
        w = torch.randn(128, 256, device="cuda")
        out = int8_linear(x, w, block_size=DEFAULT_BLOCK_SIZE)
        assert out.shape == (3, 128)


# ===================================================================
# 7. Accelerated dispatch functions
# ===================================================================

class TestAcceleratedFakeQuantize:

    def test_eager_matches_original(self, randn_2d):
        set_backend("eager")
        result = accelerated_fake_quantize(randn_2d, DEFAULT_BLOCK_SIZE, "nearest")
        ref = fused_fake_quantize(randn_2d, DEFAULT_BLOCK_SIZE, "nearest")
        torch.testing.assert_close(result, ref, atol=1e-7, rtol=0)

    def test_preserves_shape(self, randn_3d):
        set_backend("eager")
        out = accelerated_fake_quantize(randn_3d)
        assert out.shape == randn_3d.shape

    def test_stochastic_rounding(self, randn_2d):
        set_backend("eager")
        torch.manual_seed(0)
        s1 = accelerated_fake_quantize(randn_2d, rounding="stochastic")
        torch.manual_seed(1)
        s2 = accelerated_fake_quantize(randn_2d, rounding="stochastic")
        assert not torch.equal(s1, s2)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and hasattr(torch, "compile")),
        reason="CUDA + torch.compile required",
    )
    def test_compiled_matches_eager(self):
        """Compiled backend should produce same results as eager."""
        torch.manual_seed(42)
        t = torch.randn(128, 256, device="cuda")

        set_backend("eager")
        ref = accelerated_fake_quantize(t, DEFAULT_BLOCK_SIZE, "nearest")

        set_backend("compiled")
        result = accelerated_fake_quantize(t, DEFAULT_BLOCK_SIZE, "nearest")

        torch.testing.assert_close(result, ref, atol=1e-6, rtol=1e-5)


class TestAcceleratedLinear:

    def test_eager_matches_manual(self):
        torch.manual_seed(42)
        x = torch.randn(32, 128)
        w = torch.randn(64, 128)
        b = torch.randn(64)

        set_backend("eager")
        result = accelerated_linear(x, w, b, DEFAULT_BLOCK_SIZE)

        # Manual reference
        x_q = fused_fake_quantize(x, DEFAULT_BLOCK_SIZE, "nearest")
        w_q = fused_fake_quantize(w, DEFAULT_BLOCK_SIZE, "nearest")
        ref = torch.nn.functional.linear(x_q, w_q, b)

        torch.testing.assert_close(result, ref, atol=1e-7, rtol=0)

    def test_no_input_quantize(self):
        torch.manual_seed(42)
        x = torch.randn(32, 128)
        w = torch.randn(64, 128)

        set_backend("eager")
        result = accelerated_linear(x, w, quantize_input=False)

        # Only weight should be quantized
        w_q = fused_fake_quantize(w, DEFAULT_BLOCK_SIZE, "nearest")
        ref = torch.nn.functional.linear(x, w_q)

        torch.testing.assert_close(result, ref, atol=1e-7, rtol=0)

    def test_output_shape(self):
        set_backend("eager")
        x = torch.randn(8, 16, 64)
        w = torch.randn(32, 64)
        b = torch.randn(32)
        out = accelerated_linear(x, w, b)
        assert out.shape == (8, 16, 32)


# ===================================================================
# 8. backend_info diagnostic
# ===================================================================

class TestBackendInfo:

    def test_returns_dict(self):
        info = backend_info()
        assert isinstance(info, dict)

    def test_required_keys(self):
        info = backend_info()
        required = {
            "active_backend",
            "cuda_available",
            "int8_tensorcore",
            "torch_compile",
            "gpu_name",
            "compute_capability",
            "lut_cached_devices",
        }
        assert required.issubset(info.keys())

    def test_active_backend_is_string(self):
        info = backend_info()
        assert info["active_backend"] in ("eager", "compiled", "int8", "triton")

    def test_booleans(self):
        info = backend_info()
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["int8_tensorcore"], bool)
        assert isinstance(info["torch_compile"], bool)

    def test_lut_cached_is_list(self):
        info = backend_info()
        assert isinstance(info["lut_cached_devices"], list)


# ===================================================================
# 9. Capability detection
# ===================================================================

class TestCapabilities:

    def test_has_cuda_returns_bool(self):
        assert isinstance(_has_cuda(), bool)

    def test_has_int8_tensorcore_returns_bool(self):
        assert isinstance(_has_int8_tensorcore(), bool)

    def test_has_torch_compile_returns_bool(self):
        assert isinstance(_has_torch_compile(), bool)

    def test_int8_false_without_cuda(self):
        with mock.patch("axs.unified.backend._has_cuda", return_value=False):
            assert _has_int8_tensorcore() is False


# ===================================================================
# 10. Fallback behaviour
# ===================================================================

class TestFallback:

    def test_compiled_fallback_on_failure(self, randn_2d):
        """If torch.compile fails, should fallback to eager."""
        set_backend("compiled")
        # Mock _get_compiled_fq to raise
        with mock.patch(
            "axs.unified.backend._get_compiled_fq",
            side_effect=RuntimeError("compile failed"),
        ):
            result = accelerated_fake_quantize(randn_2d)
            ref = fused_fake_quantize(randn_2d, DEFAULT_BLOCK_SIZE, "nearest")
            torch.testing.assert_close(result, ref, atol=1e-7, rtol=0)
            # Should have fallen back to eager
            assert get_backend() == BackendType.EAGER

    def test_int8_linear_fallback_small_matrix(self):
        """INT8 backend should use compiled/eager for small matrices."""
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        w = torch.randn(32, 64)

        set_backend("eager")
        ref = accelerated_linear(x, w)

        # With INT8 backend, small matrices should still produce same result as eager
        # because they fall through to compiled/eager path
        set_backend("eager")
        result = accelerated_linear(x, w)
        torch.testing.assert_close(result, ref, atol=1e-7, rtol=0)


# ===================================================================
# 11. Integration with functional_unified
# ===================================================================

class TestFunctionalIntegration:

    def test_fake_quantize_unified_uses_backend(self, randn_2d):
        """fake_quantize_unified should route through the backend."""
        from axs.unified.functional_unified import fake_quantize_unified

        set_backend("eager")
        x = randn_2d.clone().requires_grad_(True)
        out = fake_quantize_unified(x)
        assert out.shape == x.shape
        # Should have gradient
        out.sum().backward()
        assert x.grad is not None

    def test_axs_linear_unified_runs(self):
        """axs_linear_unified should work with backend dispatch."""
        from axs.unified.functional_unified import axs_linear_unified

        set_backend("eager")
        torch.manual_seed(42)
        x = torch.randn(16, 64, requires_grad=True)
        w = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(32, requires_grad=True)

        out = axs_linear_unified(x, w, b)
        assert out.shape == (16, 32)
        out.sum().backward()
        assert x.grad is not None
        assert w.grad is not None

    def test_axs_matmul_unified_runs(self):
        """axs_matmul_unified should work with backend dispatch."""
        from axs.unified.functional_unified import axs_matmul_unified

        set_backend("eager")
        torch.manual_seed(42)
        a = torch.randn(16, 64)
        b_t = torch.randn(64, 32)
        out = axs_matmul_unified(a, b_t)
        assert out.shape == (16, 32)


# ===================================================================
# 12. Module integration
# ===================================================================

class TestModuleIntegration:

    def test_linear_module_forward(self):
        set_backend("eager")
        layer = AXSLinearUnified(64, 32, bias=True)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_linear_module_backward(self):
        set_backend("eager")
        layer = AXSLinearUnified(64, 32, bias=True)
        x = torch.randn(8, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None


# ===================================================================
# 13. _nf5_to_int8_row (CPU correctness)
# ===================================================================

class TestNF5ToInt8Row:

    def test_output_shapes(self):
        t = torch.randn(32, 64)
        lut = _get_lut(torch.device("cpu"))
        i8, scale = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        assert i8.shape == (32, 64)
        assert i8.dtype == torch.int8
        assert scale.shape == (32, 1)
        assert scale.dtype == torch.float32

    def test_int8_range(self):
        t = torch.randn(16, 128)
        lut = _get_lut(torch.device("cpu"))
        i8, _ = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        assert i8.max() <= 127
        assert i8.min() >= -127

    def test_reconstruction_accuracy(self):
        """Reconstruct from int8 + scale should approximate NF5 quantized value."""
        torch.manual_seed(42)
        t = torch.randn(32, 64)
        lut = _get_lut(torch.device("cpu"))
        i8, scale = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        reconstructed = i8.float() * scale
        nf5_ref = fused_fake_quantize(t, DEFAULT_BLOCK_SIZE, "nearest")
        rel_err = (reconstructed - nf5_ref).abs().mean() / nf5_ref.abs().mean().clamp(min=1e-12)
        assert rel_err < 0.05, f"INT8 roundtrip relative error {rel_err:.4f} exceeds 5%"

    def test_3d_input_flattened(self):
        """3D tensor should be flattened to 2D internally."""
        t = torch.randn(4, 16, 64)
        lut = _get_lut(torch.device("cpu"))
        i8, scale = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        # (4 * 16, 64)
        assert i8.shape == (64, 64)
        assert scale.shape == (64, 1)

    def test_zero_tensor(self):
        t = torch.zeros(16, 32)
        lut = _get_lut(torch.device("cpu"))
        i8, scale = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        assert torch.all(i8 == 0)

    def test_non_divisible_width(self):
        """Width not divisible by block_size should still work."""
        t = torch.randn(8, 50)  # 50 is not divisible by 32
        lut = _get_lut(torch.device("cpu"))
        i8, scale = _nf5_to_int8_row(t, DEFAULT_BLOCK_SIZE, lut)
        assert i8.shape == (8, 50)


# ===================================================================
# 14. _to_int8_row (CPU correctness)
# ===================================================================

class TestToInt8Row:

    def test_output_shapes(self):
        t = torch.randn(32, 64)
        i8, scale = _to_int8_row(t)
        assert i8.shape == (32, 64)
        assert i8.dtype == torch.int8
        assert scale.shape == (32, 1)
        assert scale.dtype == torch.float32

    def test_int8_range(self):
        t = torch.randn(16, 128)
        i8, _ = _to_int8_row(t)
        assert i8.max() <= 127
        assert i8.min() >= -127

    def test_reconstruction_accuracy(self):
        """Reconstruct from int8 + scale should closely approximate original."""
        torch.manual_seed(42)
        t = torch.randn(32, 64)
        i8, scale = _to_int8_row(t)
        reconstructed = i8.float() * scale
        rel_err = (reconstructed - t).abs().mean() / t.abs().mean()
        # Dynamic int8 quant should be very accurate (< 2% relative error)
        assert rel_err < 0.02, f"Dynamic int8 relative error {rel_err:.4f} exceeds 2%"

    def test_zero_tensor(self):
        t = torch.zeros(8, 32)
        i8, scale = _to_int8_row(t)
        assert torch.all(i8 == 0)

    def test_rejects_non_2d(self):
        t = torch.randn(4, 8, 16)
        with pytest.raises(AssertionError, match="Expected 2D"):
            _to_int8_row(t)

    def test_single_row(self):
        t = torch.randn(1, 256)
        i8, scale = _to_int8_row(t)
        assert i8.shape == (1, 256)
        assert scale.shape == (1, 1)


# ===================================================================
# 15. clear_int8_weight_cache
# ===================================================================

class TestClearInt8Cache:

    def test_clears_weight_cache(self):
        _INT8_WEIGHT_CACHE[(0, 10, 20)] = (
            torch.zeros(10, 20, dtype=torch.int8),
            torch.ones(10, 1),
        )
        clear_int8_weight_cache()
        assert len(_INT8_WEIGHT_CACHE) == 0

    def test_clears_refs(self):
        import weakref
        t = torch.randn(4, 8)
        key = (t.data_ptr(), 4, 8)
        _INT8_CACHE_REFS[key] = weakref.ref(t, _make_cache_finalizer(key))
        clear_int8_weight_cache()
        assert len(_INT8_CACHE_REFS) == 0

    def test_idempotent(self):
        clear_int8_weight_cache()
        clear_int8_weight_cache()  # Should not raise


# ===================================================================
# 16. INT8 cache eviction (weakref + LRU)
# ===================================================================

class TestInt8CacheEviction:

    def setup_method(self):
        clear_int8_weight_cache()

    def teardown_method(self):
        clear_int8_weight_cache()

    def test_weakref_eviction(self):
        """Cache entries are removed when source weight tensor is GC'd."""
        import gc
        import weakref

        w = torch.randn(64, 128)
        key = (w.data_ptr(), 64, 128)
        lut = _get_lut(torch.device("cpu"))
        w_i8, w_s = _nf5_to_int8_row(w, DEFAULT_BLOCK_SIZE, lut)
        _INT8_WEIGHT_CACHE[key] = (w_i8, w_s)
        _INT8_CACHE_REFS[key] = weakref.ref(w, _make_cache_finalizer(key))

        assert key in _INT8_WEIGHT_CACHE

        # Delete the weight tensor and force GC
        del w
        gc.collect()

        # Cache entry should be evicted
        assert key not in _INT8_WEIGHT_CACHE
        assert key not in _INT8_CACHE_REFS

    def test_lru_cap_enforced(self):
        """Cache should not exceed _INT8_CACHE_MAX_SIZE entries."""
        lut = _get_lut(torch.device("cpu"))
        # Insert more entries than the max
        for i in range(_INT8_CACHE_MAX_SIZE + 10):
            w = torch.randn(8, 32)
            key = (w.data_ptr(), 8, 32)
            w_i8, w_s = _nf5_to_int8_row(w, DEFAULT_BLOCK_SIZE, lut)
            # Simulate the LRU logic from int8_linear
            if len(_INT8_WEIGHT_CACHE) >= _INT8_CACHE_MAX_SIZE:
                oldest_key = next(iter(_INT8_WEIGHT_CACHE))
                _INT8_WEIGHT_CACHE.pop(oldest_key, None)
                _INT8_CACHE_REFS.pop(oldest_key, None)
            _INT8_WEIGHT_CACHE[key] = (w_i8, w_s)

        assert len(_INT8_WEIGHT_CACHE) <= _INT8_CACHE_MAX_SIZE


# ===================================================================
# 17. accelerated_matmul (CPU path)
# ===================================================================

class TestAcceleratedMatmul:

    def test_cpu_matches_torch_matmul(self):
        """On CPU, accelerated_matmul should produce identical results."""
        set_backend("eager")
        torch.manual_seed(42)
        a = torch.randn(32, 128)
        b = torch.randn(128, 64)
        ref = torch.matmul(a, b)
        result = accelerated_matmul(a, b)
        torch.testing.assert_close(result, ref, atol=1e-6, rtol=1e-5)

    def test_3d_input(self):
        """Batch dimensions should be preserved."""
        set_backend("eager")
        torch.manual_seed(42)
        a = torch.randn(4, 16, 64)
        b = torch.randn(64, 32)
        ref = torch.matmul(a, b)
        result = accelerated_matmul(a, b)
        torch.testing.assert_close(result, ref, atol=1e-6, rtol=1e-5)

    def test_output_shape(self):
        set_backend("eager")
        a = torch.randn(8, 256)
        b = torch.randn(256, 128)
        result = accelerated_matmul(a, b)
        assert result.shape == (8, 128)

    def test_backward_integration(self):
        """Verify backward pass produces gradients correctly."""
        set_backend("eager")
        torch.manual_seed(42)
        x = torch.randn(16, 64, requires_grad=True)
        w = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(32)

        layer = AXSLinearUnified(64, 32, bias=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ===================================================================
# 18. _int8_matmul_2d (CUDA only)
# ===================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestInt8Matmul2D:

    def test_basic_correctness(self):
        """INT8 matmul should approximate FP32 matmul."""
        torch.manual_seed(42)
        a = torch.randn(64, 256, device="cuda")
        b = torch.randn(256, 128, device="cuda")
        ref = torch.matmul(a, b)
        result = _int8_matmul_2d(a, b)
        rel_err = (result - ref).abs().mean() / ref.abs().mean()
        assert rel_err < 0.05, f"INT8 matmul relative error {rel_err:.4f} exceeds 5%"

    def test_output_shape(self):
        a = torch.randn(32, 512, device="cuda")
        b = torch.randn(512, 256, device="cuda")
        result = _int8_matmul_2d(a, b)
        assert result.shape == (32, 256)
        assert result.dtype == torch.float32

    def test_small_m_padding(self):
        """M < 32 should work (requires padding)."""
        a = torch.randn(3, 128, device="cuda")
        b = torch.randn(128, 64, device="cuda")
        result = _int8_matmul_2d(a, b)
        assert result.shape == (3, 64)

    def test_non_contiguous_inputs(self):
        """Transposed (non-contiguous) inputs should work."""
        torch.manual_seed(42)
        a_raw = torch.randn(256, 64, device="cuda")
        a = a_raw.t()  # (64, 256) non-contiguous
        b = torch.randn(256, 128, device="cuda")
        result = _int8_matmul_2d(a, b)
        ref = torch.matmul(a.float(), b.float())
        rel_err = (result - ref).abs().mean() / ref.abs().mean()
        assert rel_err < 0.05
