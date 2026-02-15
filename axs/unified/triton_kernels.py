"""
AXS-6 Triton Kernels — Fused NF5 Fake-Quantize
================================================

Hand-written Triton kernel for the NF5 fused fake-quantize hot path.
Achieves **15×** speedup over eager PyTorch and **9×** over torch.compile
by processing multiple quantisation blocks in parallel via a 2-D tile
strategy.

Key design choices
------------------
- **2-D vectorisation**: each Triton program processes ``N_BLOCKS`` blocks
  of ``BLOCK_SIZE`` elements simultaneously, keeping the GPU fully
  occupied with a compact grid.
- **Bit-exact scale**: a ``tl.where(scale <= amax, scale * 2, scale)``
  guard corrects the ±1 ULP fast-math imprecision of
  ``floor(log2(amax))`` near exact powers of two.
- **Optional stochastic rounding**: pass ``STOCHASTIC=True`` to add
  uniform dither of ±0.5 LUT steps before truncation, matching the
  ``rounding="stochastic"`` mode of :func:`fused_fake_quantize`.

RTX 5070 Ti, 4096×4096 float32 (measured, 100 iterations):

    ============  ========  ==========
    Backend        ms/call   vs Eager
    ============  ========  ==========
    **Triton**     0.36 ms   **15×**
    Compiled       3.29 ms   1.7×
    Eager          5.47 ms   1.0×
    ============  ========  ==========
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

from axs.unified.quantize_unified import (
    _LUT_MAX_IDX,
    DEFAULT_BLOCK_SIZE,
    FUSED_NF5_LUT,
    VALID_BLOCK_SIZES,
)

# ── Triton availability check ──────────────────────────────────────────────

def has_triton() -> bool:
    """Return True if Triton is importable *and* a CUDA device is available."""
    return _HAS_TRITON and torch.cuda.is_available()


# ── LUT cache (one per device) ─────────────────────────────────────────────

_LUT_CACHE: dict[int, torch.Tensor] = {}


def _get_lut(device: torch.device) -> torch.Tensor:
    """Return the NF5 fused LUT on *device*, caching across calls."""
    dev_idx = device.index if device.index is not None else 0
    if dev_idx not in _LUT_CACHE:
        _LUT_CACHE[dev_idx] = FUSED_NF5_LUT.to(device)
    return _LUT_CACHE[dev_idx]


# ── Kernel ──────────────────────────────────────────────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _nf5_fq_kernel(
        X_ptr,
        OUT_ptr,
        LUT_ptr,
        total_blocks,
        seed,
        BLOCK_SIZE: tl.constexpr,
        N_BLOCKS: tl.constexpr,
        LUT_MAX: tl.constexpr,
        STOCHASTIC: tl.constexpr,
    ):
        """Fused NF5 fake-quantize — processes N_BLOCKS blocks per program.

        Each program loads a 2-D tile of shape ``(N_BLOCKS, BLOCK_SIZE)``
        and performs sign extraction, power-of-2 scaling, LUT gather, and
        denormalisation in a single pass.
        """
        pid = tl.program_id(0)
        block_start = pid * N_BLOCKS

        # 2-D offsets: rows = blocks, cols = elements within a block
        block_offs = tl.arange(0, N_BLOCKS)
        elem_offs = tl.arange(0, BLOCK_SIZE)
        flat_idx = (
            (block_start + block_offs[:, None]) * BLOCK_SIZE
            + elem_offs[None, :]
        )
        mask = (block_start + block_offs[:, None]) < total_blocks

        # Load tile ──────────────────────────────────────────────────────
        x = tl.load(X_ptr + flat_idx, mask=mask, other=0.0).to(tl.float32)

        # Sign (0 for exact zeros, matching torch.sign) ──────────────────
        signs = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
        abs_vals = tl.abs(x)

        # Per-block amax → power-of-2 scale ──────────────────────────────
        amax = tl.max(abs_vals, axis=1)[:, None]  # (N_BLOCKS, 1)
        safe_amax = tl.maximum(amax, 1e-12)
        log2_val = tl.math.log2(safe_amax)
        floor_val = tl.math.floor(log2_val)
        scale = tl.math.exp2(floor_val + 1.0)
        # Guard against fast-math undershoot at exact powers of two
        scale = tl.where(scale <= amax, scale * 2.0, scale)
        scale = tl.where(amax == 0.0, 1.0, scale)

        # Normalise to [0, 1] ────────────────────────────────────────────
        normalised = abs_vals / scale
        normalised = tl.minimum(normalised, 1.0)
        normalised = tl.maximum(normalised, 0.0)

        # LUT index ──────────────────────────────────────────────────────
        if STOCHASTIC:
            # Uniform dither of ±0.5 LUT steps (matches PyTorch path)
            rand_vals = tl.rand(seed, flat_idx)  # uniform [0, 1)
            dither = (rand_vals - 0.5) / LUT_MAX
            normalised = normalised + dither
            normalised = tl.minimum(normalised, 1.0)
            normalised = tl.maximum(normalised, 0.0)

        idx = (normalised * LUT_MAX).to(tl.int32)
        idx = tl.minimum(idx, LUT_MAX)
        idx = tl.maximum(idx, 0)

        # Gather from LUT and denormalise ─────────────────────────────────
        recon = tl.load(LUT_ptr + idx, mask=mask, other=0.0)
        result = signs * recon * scale
        tl.store(OUT_ptr + flat_idx, result, mask=mask)

    # ── FQ tile helper (called from fused matmul kernel) ────────────────

    @triton.jit
    def _nf5_fq_tile_2d(
        tile,
        LUT_ptr,
        LUT_MAX: tl.constexpr,
    ):
        """Apply NF5 fake-quantize per row of a 2-D register tile.

        Each row is one quantisation block.  Computes per-row amax,
        power-of-2 scale, normalise, LUT gather, and denorm — entirely
        in registers with no global-memory round-trip.
        """
        signs = tl.where(tile > 0, 1.0, tl.where(tile < 0, -1.0, 0.0))
        abs_vals = tl.abs(tile)
        amax = tl.max(abs_vals, axis=1)[:, None]
        safe_amax = tl.maximum(amax, 1e-12)
        log2_val = tl.math.log2(safe_amax)
        floor_val = tl.math.floor(log2_val)
        scale = tl.math.exp2(floor_val + 1.0)
        # Guard against fast-math undershoot near exact powers of two
        scale = tl.where(scale <= amax, scale * 2.0, scale)
        scale = tl.where(amax == 0.0, 1.0, scale)
        normalised = abs_vals / scale
        normalised = tl.minimum(normalised, 1.0)
        normalised = tl.maximum(normalised, 0.0)
        idx = (normalised * LUT_MAX).to(tl.int32)
        idx = tl.minimum(idx, LUT_MAX)
        idx = tl.maximum(idx, 0)
        recon = tl.load(LUT_ptr + idx)
        return signs * recon * scale

    # ── Fused FQ + matmul kernel ────────────────────────────────────────

    @triton.jit
    def _nf5_fq_linear_kernel(
        X_ptr,
        W_ptr,
        OUT_ptr,
        BIAS_ptr,
        LUT_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        LUT_MAX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        QUANTIZE_X: tl.constexpr,
        USE_TF32: tl.constexpr,
    ):
        """Fused NF5 fake-quantize + matmul in a single kernel.

        Computes ``output = FQ(X) @ FQ(W).T [+ bias]`` where the NF5
        fake-quantisation is applied on-the-fly to each K-tile in
        registers.  Intermediate quantised tensors are never written
        to global memory.

        ``BLOCK_K`` must equal the NF5 block_size (32) so that each
        K-tile row is exactly one quantisation block.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        # Grouped ordering for better L2 cache reuse
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # ── Load + FQ input tile [BLOCK_M, BLOCK_K] ──
            x_ptrs = (
                X_ptr
                + offs_m[:, None] * stride_xm
                + offs_k[None, :] * stride_xk
            )
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            if QUANTIZE_X:
                x = _nf5_fq_tile_2d(x, LUT_ptr, LUT_MAX)

            # ── Load + FQ weight tile [BLOCK_N, BLOCK_K] ──
            w_ptrs = (
                W_ptr
                + offs_n[:, None] * stride_wn
                + offs_k[None, :] * stride_wk
            )
            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

            w = _nf5_fq_tile_2d(w, LUT_ptr, LUT_MAX)

            # ── Accumulate: [BM, BK] @ [BK, BN] → [BM, BN] ──
            if USE_TF32:
                acc += tl.dot(x, tl.trans(w), input_precision="tf32")
            else:
                acc += tl.dot(x, tl.trans(w), input_precision="ieee")

        # Bias
        if HAS_BIAS:
            bias = tl.load(
                BIAS_ptr + offs_n, mask=offs_n < N, other=0.0
            )
            acc += bias[None, :]

        # Store output tile
        out_ptrs = (
            OUT_ptr
            + offs_m[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)


# ── Default tile size ───────────────────────────────────────────────────────

# N_BLOCKS=32 is generally optimal: produces enough programs for full GPU
# occupancy while keeping register pressure low.  Adjust via the
# ``n_blocks`` parameter of :func:`triton_fused_fake_quantize` if needed.
_DEFAULT_N_BLOCKS: int = 32


# ── Public API ──────────────────────────────────────────────────────────────

def triton_fused_fake_quantize(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
    *,
    n_blocks: int = _DEFAULT_N_BLOCKS,
) -> torch.Tensor:
    """Triton-accelerated NF5 fused fake-quantize.

    Drop-in replacement for :func:`quantize_unified.fused_fake_quantize`
    with identical numerics (bit-exact for ``rounding='nearest'``).

    Supports FP32, BF16, and FP16 inputs.  The NF5 computation is always
    performed in FP32 internally; the output is cast back to the input
    dtype.

    Args:
        tensor: Input tensor of any shape on a CUDA device.
        block_size: Quantisation block size (8, 16, or 32).
        rounding: ``'nearest'`` or ``'stochastic'``.
        n_blocks: Number of blocks processed per Triton program (tuning knob).

    Returns:
        Tensor of the same shape **and dtype** with AXS-6 NF5 quantisation
        noise.

    Raises:
        RuntimeError: If Triton is not available or tensor is not on CUDA.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not tensor.is_cuda:
        raise RuntimeError("triton_fused_fake_quantize requires a CUDA tensor")
    assert block_size in VALID_BLOCK_SIZES, (
        f"block_size must be one of {VALID_BLOCK_SIZES}"
    )

    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    device = tensor.device
    lut = _get_lut(device)

    # Reshape to (-1, last_dim), pad last dim — matches fused_fake_quantize
    # Always compute in FP32 (LUT is FP32); cast back at the end.
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    last_dim = flat.shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    flat_1d = flat.reshape(-1).contiguous()
    total = flat_1d.numel() // block_size
    n_progs = (total + n_blocks - 1) // n_blocks
    out = torch.empty_like(flat_1d)

    stochastic = rounding == "stochastic"
    seed = torch.randint(0, 2**31, (1,), device=device).item() if stochastic else 0

    _nf5_fq_kernel[(n_progs,)](
        flat_1d,
        out,
        lut,
        total,
        seed,
        BLOCK_SIZE=block_size,
        N_BLOCKS=n_blocks,
        LUT_MAX=_LUT_MAX_IDX,
        STOCHASTIC=stochastic,
    )

    # Un-pad and restore original shape and dtype
    out = out.reshape(flat.shape)
    if pad > 0:
        out = out[:, :last_dim]
    result = out.reshape(orig_shape)
    return result if orig_dtype == torch.float32 else result.to(orig_dtype)


# ── Fused linear defaults ──────────────────────────────────────────────────

_DEFAULT_FUSED_BLOCK_M: int = 64
_DEFAULT_FUSED_BLOCK_N: int = 64
_DEFAULT_FUSED_GROUP_M: int = 8

# Size threshold: fused kernel is faster than separate FQ + cuBLAS only
# for small matrices where kernel-launch overhead dominates.
# Above this output-element count, fall back to the separate path.
_FUSED_SIZE_THRESHOLD: int = 128 * 1024  # ~128K output elements


def triton_fused_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    quantize_input: bool = True,
) -> torch.Tensor:
    """Fused NF5 fake-quantize + linear via a custom Triton kernel.

    Computes ``output = FQ(input) @ FQ(weight).T + bias`` in a **single
    kernel launch**.  The NF5 quantisation is applied on-the-fly to each
    K-tile in GPU registers — intermediate quantised tensors are never
    materialised, saving memory bandwidth and kernel-launch overhead.

    .. note::

        Currently requires ``block_size=32`` (the default) because the
        matmul tile width (``BLOCK_K``) must exactly equal the NF5 block
        size so each tile row spans one quantisation block.  Other block
        sizes fall back to the separate ``FQ + F.linear`` path.

    Args:
        input: Input tensor ``(*, in_features)`` on CUDA.
        weight: Weight matrix ``(out_features, in_features)`` on CUDA.
        bias: Optional bias ``(out_features,)``.
        block_size: NF5 block size — must be 32 for the fused kernel.
        quantize_input: Whether to also quantise the input activations.

    Returns:
        Output tensor ``(*, out_features)``, float32.

    Raises:
        RuntimeError: If Triton is not available or tensors are not CUDA.
        ValueError: If ``block_size != 32``.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not input.is_cuda:
        raise RuntimeError("triton_fused_linear requires CUDA tensors")
    if block_size != 32:
        raise ValueError(
            f"triton_fused_linear requires block_size=32, got {block_size}. "
            "Use the separate FQ + F.linear path for other block sizes."
        )

    # Flatten input to 2-D (always compute in FP32; cast output back)
    orig_shape = input.shape
    orig_dtype = input.dtype
    x_2d = input.reshape(-1, input.shape[-1]).contiguous().float()
    M, K = x_2d.shape
    N = weight.shape[0]

    w = weight.contiguous().float()

    # Output buffer
    out = torch.empty((M, N), device=input.device, dtype=torch.float32)

    # LUT
    lut = _get_lut(input.device)

    # Tile sizes (BLOCK_K = block_size for quantisation alignment)
    BLOCK_M = _DEFAULT_FUSED_BLOCK_M
    BLOCK_N = _DEFAULT_FUSED_BLOCK_N
    BLOCK_K = block_size  # 32
    GROUP_M = _DEFAULT_FUSED_GROUP_M

    # Grid: one program per (BLOCK_M × BLOCK_N) output tile
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _nf5_fq_linear_kernel[grid](
        x_2d,
        w,
        out,
        bias if bias is not None else out,  # dummy ptr (never loaded)
        lut,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        LUT_MAX=_LUT_MAX_IDX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        HAS_BIAS=bias is not None,
        QUANTIZE_X=quantize_input,
        USE_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    # Restore original batch dimensions and dtype
    out_shape = orig_shape[:-1] + (N,)
    result = out.reshape(out_shape)
    return result if orig_dtype == torch.float32 else result.to(orig_dtype)
