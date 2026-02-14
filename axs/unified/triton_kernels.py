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

    Args:
        tensor: Input tensor of any shape on a CUDA device.
        block_size: Quantisation block size (8, 16, or 32).
        rounding: ``'nearest'`` or ``'stochastic'``.
        n_blocks: Number of blocks processed per Triton program (tuning knob).

    Returns:
        Float32 tensor of the same shape with AXS-6 NF5 quantisation noise.

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
    device = tensor.device
    lut = _get_lut(device)

    # Reshape to (-1, last_dim), pad last dim — matches fused_fake_quantize
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

    # Un-pad and restore original shape
    out = out.reshape(flat.shape)
    if pad > 0:
        out = out[:, :last_dim]
    return out.reshape(orig_shape)
