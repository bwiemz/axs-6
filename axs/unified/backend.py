"""
AXS-6 Hardware Backend Dispatch
=================================

Automatically selects the fastest available backend for AXS-6 operations:

  1. **Triton** — Hand-written Triton kernel that fuses the entire NF5 LUT
     fake-quantize into a single pass.  ~15× faster than eager.  Requires
     Triton ≥ 3.0 and a CUDA GPU.
  2. **Compiled** — ``torch.compile`` fuses NF5 LUT ops into efficient Triton
     kernels.  ~2–4× faster than eager.  Requires PyTorch 2.1+ with Triton.
  3. **INT8 Tensor Core** — For very large matmuls, encodes NF5 values to int8
     and uses ``torch._int_mm`` for the GEMM, with per-row scale correction.
     Requires CUDA compute capability ≥ 7.5 (Turing+).
  4. **Eager** — Pure-PyTorch fallback.  Always works.

The backend is selected automatically at first use and cached.  Users can
override with :func:`set_backend` or the ``AXS6_BACKEND`` environment variable.

Usage::

    from axs.unified.backend import get_backend, set_backend, BackendType

    # Auto-detect (default)
    backend = get_backend()
    print(backend.name)  # "triton" on most CUDA setups

    # Force a specific backend
    set_backend("eager")

    # Accelerated fake-quantize (uses whatever backend is active)
    from axs.unified.backend import accelerated_fake_quantize
    x_q = accelerated_fake_quantize(x, block_size=32)

    # Accelerated linear (uses INT8 matmul when beneficial)
    from axs.unified.backend import accelerated_linear
    y = accelerated_linear(x, weight, bias, block_size=32)
"""

from __future__ import annotations

import enum
import functools
import logging
import os
from typing import Literal

import torch
import torch.nn.functional as F

from axs.core import DEFAULT_BLOCK_SIZE
from axs.unified.quantize_unified import (
    _LUT_MAX_IDX,
    FUSED_NF5_LUT,
    fused_fake_quantize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device-cached LUT (avoids .to(device) in the hot path)
# ---------------------------------------------------------------------------

_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_lut(device: torch.device) -> torch.Tensor:
    """Return the NF5 LUT on the requested device, caching the result."""
    if device not in _LUT_CACHE:
        _LUT_CACHE[device] = FUSED_NF5_LUT.to(device)
    return _LUT_CACHE[device]


# ---------------------------------------------------------------------------
# Backend types
# ---------------------------------------------------------------------------

class BackendType(enum.Enum):
    """Available AXS-6 compute backends."""

    EAGER = "eager"
    COMPILED = "compiled"
    INT8 = "int8"
    TRITON = "triton"


# ---------------------------------------------------------------------------
# Backend capability detection
# ---------------------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_int8_tensorcore() -> bool:
    """INT8 tensor cores available on Turing+ (sm_75+)."""
    if not _has_cuda():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 7 or (cap[0] == 7 and cap[1] >= 5)


def _has_torch_compile() -> bool:
    """Check if torch.compile is available and functional."""
    import importlib

    if not hasattr(torch, "compile"):
        return False
    # torch.compile needs inductor backend + triton
    try:
        importlib.import_module("torch._inductor")
        return True
    except ImportError:
        return False


def _has_triton_kernel() -> bool:
    """Check if the custom Triton NF5 kernel is available."""
    try:
        from axs.unified.triton_kernels import has_triton

        return has_triton()
    except ImportError:
        return False


def detect_best_backend() -> BackendType:
    """Auto-detect the best available backend."""
    # Environment override
    env = os.environ.get("AXS6_BACKEND", "").lower().strip()
    if env in ("eager", "compiled", "int8", "triton"):
        return BackendType(env)

    # Prefer: triton > compiled > eager
    if _has_triton_kernel():
        return BackendType.TRITON
    if _has_cuda() and _has_torch_compile():
        return BackendType.COMPILED
    return BackendType.EAGER


# ---------------------------------------------------------------------------
# Global backend state
# ---------------------------------------------------------------------------

_active_backend: BackendType | None = None
_compiled_fq: callable | None = None
_compiled_fq_stochastic: callable | None = None


def get_backend() -> BackendType:
    """Return the currently active backend, auto-detecting if needed."""
    global _active_backend
    if _active_backend is None:
        _active_backend = detect_best_backend()
        logger.info("AXS-6 backend: %s", _active_backend.value)
    return _active_backend


def set_backend(backend: str | BackendType) -> None:
    """
    Override the active backend.

    Args:
        backend: One of ``"eager"``, ``"compiled"``, ``"int8"``, or
            ``"triton"``.
    """
    global _active_backend, _compiled_fq, _compiled_fq_stochastic
    if isinstance(backend, str):
        backend = BackendType(backend.lower())
    _active_backend = backend
    # Reset compiled caches when switching
    _compiled_fq = None
    _compiled_fq_stochastic = None
    logger.info("AXS-6 backend set to: %s", backend.value)


# ---------------------------------------------------------------------------
# Compiled backend: torch.compile'd fake-quantize
# ---------------------------------------------------------------------------

def _fused_fq_compilable(
    tensor: torch.Tensor,
    block_size: int,
    stochastic: bool,
) -> torch.Tensor:
    """
    Compilable version of fused_fake_quantize.

    Identical logic to :func:`fused_fake_quantize` but avoids the
    ``FUSED_NF5_LUT.to(device)`` call in the hot path (which breaks
    CUDA graph capture).
    """
    original_shape = tensor.shape
    device = tensor.device
    orig_dtype = tensor.dtype
    lut = _get_lut(device)

    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    last_dim = flat.shape[-1]
    pad_amount = (block_size - last_dim % block_size) % block_size
    if pad_amount > 0:
        flat = F.pad(flat, (0, pad_amount))

    num_blocks = flat.shape[-1] // block_size
    blocked = flat.reshape(-1, num_blocks, block_size)

    signs = blocked.sign()
    abs_vals = blocked.abs()

    amax = abs_vals.amax(dim=-1, keepdim=True)
    safe_amax = amax.clamp(min=1e-45)
    scales = torch.exp2(safe_amax.log2().float().floor() + 1.0)
    zero_mask = amax == 0
    scales = scales.masked_fill(zero_mask, 1.0)

    normalised = (abs_vals / scales).clamp(0.0, 1.0)

    if stochastic:
        dither = (torch.rand_like(normalised) - 0.5) / _LUT_MAX_IDX
        idx = ((normalised + dither) * _LUT_MAX_IDX).to(torch.int32).clamp(0, _LUT_MAX_IDX)
    else:
        idx = (normalised * _LUT_MAX_IDX).to(torch.int32).clamp(0, _LUT_MAX_IDX)

    reconstructed_norm = lut[idx.long()]
    result = signs * reconstructed_norm * scales
    result = result.reshape(-1, num_blocks * block_size)

    if pad_amount > 0:
        result = result[:, :last_dim]

    return result.reshape(original_shape).to(orig_dtype)


def _get_compiled_fq(stochastic: bool = False) -> callable:
    """Get or create the compiled fake-quantize function."""
    global _compiled_fq, _compiled_fq_stochastic
    if stochastic:
        if _compiled_fq_stochastic is None:
            fn = functools.partial(_fused_fq_compilable, stochastic=True)
            _compiled_fq_stochastic = torch.compile(fn, mode="default")
        return _compiled_fq_stochastic
    else:
        if _compiled_fq is None:
            fn = functools.partial(_fused_fq_compilable, stochastic=False)
            _compiled_fq = torch.compile(fn, mode="default")
        return _compiled_fq


# ---------------------------------------------------------------------------
# INT8 tensor core backend
# ---------------------------------------------------------------------------

# Cache for padded weight encodings to avoid re-encoding every call
_INT8_WEIGHT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _nf5_to_int8_row(
    tensor: torch.Tensor,
    block_size: int,
    lut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NF5 fake-quantize → per-row int8 encoding + scale.

    Handles arbitrary batch dimensions by flattening to 2D.

    Returns:
        int8_tensor: ``[rows, cols]`` int8
        row_scale: ``[rows, 1]`` float32 (multiply to recover FP32)
    """
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    rows = flat.shape[0]
    last_dim = flat.shape[-1]

    pad = (block_size - last_dim % block_size) % block_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    num_blocks = flat.shape[-1] // block_size
    blocked = flat.reshape(rows, num_blocks, block_size)

    signs = blocked.sign()
    abs_vals = blocked.abs()
    amax = abs_vals.amax(dim=-1, keepdim=True).clamp(min=1e-45)
    scales = torch.exp2(amax.log2().float().floor() + 1.0)
    scales = scales.masked_fill(amax < 1e-44, 1.0)

    normalised = (abs_vals / scales).clamp(0.0, 1.0)
    idx = (normalised * _LUT_MAX_IDX).to(torch.int32).clamp(0, _LUT_MAX_IDX)

    nf5_vals = signs * lut[idx.long()] * scales
    nf5_flat = nf5_vals.reshape(rows, -1)
    if pad > 0:
        nf5_flat = nf5_flat[:, :last_dim]

    row_amax = nf5_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    int8_vals = (nf5_flat / row_amax * 127).round().clamp(-127, 127).to(torch.int8)
    row_scale = row_amax / 127.0

    return int8_vals, row_scale


def _pad_to_multiple(tensor: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad tensor along `dim` to a multiple of `multiple`."""
    size = tensor.shape[dim]
    pad_amount = (multiple - size % multiple) % multiple
    if pad_amount == 0:
        return tensor
    pad_spec = [0] * (2 * tensor.ndim)
    # F.pad uses reversed dimension order
    pad_spec[2 * (tensor.ndim - 1 - dim) + 1] = pad_amount
    return F.pad(tensor, pad_spec)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    AXS-6 linear using INT8 tensor cores.

    Applies NF5 fake-quantization then performs the matmul via
    ``torch._int_mm`` with per-row scale correction.

    Handles arbitrary batch dimensions (not just 2D). Caches weight
    encoding by data_ptr to avoid redundant re-encoding.

    Args:
        x: Input ``(*, in_features)``.
        weight: Weight ``(out_features, in_features)``.
        bias: Optional bias.
        block_size: AXS-6 block size.

    Returns:
        Output ``(*, out_features)``.
    """
    device = x.device
    lut = _get_lut(device)

    # Flatten to 2D for int_mm
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]

    # Encode input to int8
    x_i8, x_s = _nf5_to_int8_row(x_2d, block_size, lut)

    # Cache weight encoding (weights don't change within a step)
    w_key = weight.data_ptr()
    cached = _INT8_WEIGHT_CACHE.get(w_key)
    if cached is not None and cached[0].shape == (weight.shape[0], weight.shape[1]):
        w_i8, w_s = cached
    else:
        w_i8, w_s = _nf5_to_int8_row(weight, block_size, lut)
        _INT8_WEIGHT_CACHE[w_key] = (w_i8, w_s)

    # Pad M to multiple of 32 (torch._int_mm requirement on some GPUs)
    M_pad = (32 - M % 32) % 32
    if M_pad > 0:
        x_i8 = F.pad(x_i8, (0, 0, 0, M_pad))

    # INT8 tensor core matmul
    result_int = torch._int_mm(x_i8, w_i8.t())  # [M_padded, out_f] int32

    # Scale correction
    if M_pad > 0:
        result_fp = result_int[:M].float() * x_s * w_s.t()
    else:
        result_fp = result_int.float() * x_s * w_s.t()

    if bias is not None:
        result_fp = result_fp + bias

    # Restore original batch shape
    out_shape = orig_shape[:-1] + (weight.shape[0],)
    return result_fp.reshape(out_shape)


# ---------------------------------------------------------------------------
# Accelerated dispatch functions
# ---------------------------------------------------------------------------

def accelerated_fake_quantize(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> torch.Tensor:
    """
    Fake-quantize using the best available backend.

    - **triton**: Custom Triton kernel (~15× faster than eager)
    - **compiled**: ``torch.compile``'d NF5 LUT (~2–4× faster than eager)
    - **eager**: Original pure-PyTorch implementation
    - **int8**: Falls back to compiled FQ (INT8 only helps the matmul)
    """
    backend = get_backend()

    # Triton kernel — fastest path
    if backend == BackendType.TRITON:
        try:
            from axs.unified.triton_kernels import triton_fused_fake_quantize

            return triton_fused_fake_quantize(tensor, block_size, rounding)
        except Exception:
            logger.warning("Triton kernel failed, falling back to compiled")
            set_backend("compiled")
            return accelerated_fake_quantize(tensor, block_size, rounding)

    if backend == BackendType.COMPILED or backend == BackendType.INT8:
        try:
            fn = _get_compiled_fq(stochastic=(rounding == "stochastic"))
            result = fn(tensor, block_size)
            # Safety: ensure output stays on the same device as input
            # (torch.compile can occasionally misplace tensors in
            #  custom autograd contexts)
            if result.device != tensor.device:
                result = result.to(tensor.device)
            return result
        except Exception:
            # Compilation failed — fall back gracefully
            logger.warning("torch.compile failed, falling back to eager")
            set_backend("eager")
            return fused_fake_quantize(tensor, block_size, rounding)

    return fused_fake_quantize(tensor, block_size, rounding)


def accelerated_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
) -> torch.Tensor:
    """
    AXS-6 quantised linear using the best available backend.

    - **int8**: NF5 quantize → INT8 tensor core matmul (large matrices)
    - **compiled**: NF5 quantize via torch.compile → FP32 matmul
    - **eager**: Original pure-PyTorch implementation

    For the INT8 path, both weights and activations are NF5-quantised then
    dynamically scaled to int8 for the GEMM.  The INT8 path is only used
    when both dimensions are large enough to benefit (≥ 512).
    """
    backend = get_backend()

    # INT8 path for large matrices where tensor cores help
    if backend == BackendType.INT8 and _has_int8_tensorcore():
        K = input.shape[-1]
        N = weight.shape[0]
        if K >= 512 and N >= 512:
            return int8_linear(input, weight, bias, block_size)
        # Fall through to compiled for small matrices

    # Triton path: fused kernel for small matrices, separate FQ+cuBLAS for large
    if backend == BackendType.TRITON:
        try:
            from axs.unified.triton_kernels import (
                _FUSED_SIZE_THRESHOLD,
                triton_fused_fake_quantize,
                triton_fused_linear,
            )

            # Size-based dispatch: the fused kernel keeps FQ'd values in
            # registers (no intermediate writes) but its matmul can't compete
            # with cuBLAS for large matrices.
            M_flat = input.reshape(-1, input.shape[-1]).shape[0]
            N = weight.shape[0]
            use_fused = (
                block_size == 32
                and M_flat * N <= _FUSED_SIZE_THRESHOLD
            )

            if use_fused:
                return triton_fused_linear(
                    input, weight, bias, block_size, quantize_input,
                )
            else:
                w_q = triton_fused_fake_quantize(weight, block_size, "nearest")
                x_q = (
                    triton_fused_fake_quantize(input, block_size, "nearest")
                    if quantize_input
                    else input
                )
                return F.linear(x_q, w_q, bias)
        except Exception:
            logger.warning("Triton kernel failed in linear, falling back to compiled")
            set_backend("compiled")
            return accelerated_linear(input, weight, bias, block_size, quantize_input)

    # Compiled path: fused NF5 FQ + FP32 matmul
    if backend in (BackendType.COMPILED, BackendType.INT8):
        try:
            fq = _get_compiled_fq(stochastic=False)
            w_q = fq(weight, block_size)
            x_q = fq(input, block_size) if quantize_input else input
            return F.linear(x_q, w_q, bias)
        except Exception:
            logger.warning("torch.compile failed, falling back to eager")
            set_backend("eager")

    # Eager fallback
    w_q = fused_fake_quantize(weight, block_size, "nearest")
    x_q = fused_fake_quantize(input, block_size, "nearest") if quantize_input else input
    return F.linear(x_q, w_q, bias)


# ---------------------------------------------------------------------------
# Info / diagnostic
# ---------------------------------------------------------------------------

def backend_info() -> dict[str, object]:
    """Return diagnostic info about the active backend and capabilities."""
    return {
        "active_backend": get_backend().value,
        "cuda_available": _has_cuda(),
        "triton_kernel": _has_triton_kernel(),
        "int8_tensorcore": _has_int8_tensorcore(),
        "torch_compile": _has_torch_compile(),
        "gpu_name": torch.cuda.get_device_name() if _has_cuda() else None,
        "compute_capability": (
            torch.cuda.get_device_capability() if _has_cuda() else None
        ),
        "lut_cached_devices": list(str(d) for d in _LUT_CACHE),
    }


def clear_int8_weight_cache() -> None:
    """Clear the INT8 weight encoding cache (call after optimizer.step())."""
    _INT8_WEIGHT_CACHE.clear()
