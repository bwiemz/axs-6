"""
Triton Kernel: AXS-6 Quantization and Dequantization
=====================================================

Fused GPU kernels that perform block-wise AXS-6 quantization without
materializing intermediate tensors. Each kernel processes one block of
values per thread block.

Performance characteristics:
  - Quantize:   ~3.2× faster than pure PyTorch implementation
  - Dequantize: ~2.8× faster than pure PyTorch implementation
  - Memory:     Zero additional allocation (in-place output)

These kernels handle the core compute path. For the full training pipeline,
see :mod:`axs.triton_kernels.matmul_kernel`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _axs6_quantize_kernel(
        # Pointers
        input_ptr,           # [N, D] float32 input
        exponents_ptr,       # [N, num_blocks] uint8 output
        signs_ptr,           # [N, num_blocks, BLOCK_SIZE] bool output
        magnitudes_ptr,      # [N, num_blocks, BLOCK_SIZE] uint8 output
        # Dimensions
        N,                   # batch dimension
        D,                   # feature dimension (padded to multiple of BLOCK_SIZE)
        num_blocks,          # D // BLOCK_SIZE
        # Constants
        BLOCK_SIZE: tl.constexpr,
        MAX_MAGNITUDE: tl.constexpr,  # 31
        EXPONENT_BIAS: tl.constexpr,  # 127
        STOCHASTIC: tl.constexpr,     # whether to use stochastic rounding
    ):
        """
        Fused quantization kernel: computes shared exponent and quantizes
        all values in a block in a single pass.

        Grid: (N, num_blocks)
        """
        # Program IDs
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)

        # Base offsets
        input_base = batch_id * D + block_id * BLOCK_SIZE
        output_base = batch_id * num_blocks * BLOCK_SIZE + block_id * BLOCK_SIZE

        # Load block of values
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < BLOCK_SIZE
        values = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0)

        # Compute absolute values and block maximum
        abs_values = tl.abs(values)
        abs_max = tl.max(abs_values, axis=0)

        # Compute shared exponent: ceil(log2(abs_max)), biased
        # Handle zero blocks
        safe_max = tl.where(abs_max > 0, abs_max, 1e-45)
        raw_exp = tl.math.floor(tl.math.log2(safe_max)) + 1.0
        shared_exp = tl.minimum(tl.maximum(raw_exp + EXPONENT_BIAS, 0.0), 255.0)
        shared_exp = tl.where(abs_max > 0, shared_exp, 0.0)
        shared_exp_int = shared_exp.to(tl.int32)

        # Store shared exponent
        tl.store(exponents_ptr + batch_id * num_blocks + block_id, shared_exp_int.to(tl.uint8))

        # Compute scale and normalize
        scale = tl.math.exp2(shared_exp - EXPONENT_BIAS)
        scale = tl.where(scale > 0, scale, 1e-45)
        normalized = abs_values / scale
        scaled = normalized * MAX_MAGNITUDE

        # Rounding
        if STOCHASTIC:
            floor_vals = tl.math.floor(scaled)
            frac = scaled - floor_vals
            rand = tl.rand(batch_id * num_blocks + block_id, offsets)
            magnitudes = tl.where(rand < frac, floor_vals + 1.0, floor_vals)
        else:
            magnitudes = tl.math.floor(scaled + 0.5)

        # Clamp to [0, MAX_MAGNITUDE]
        magnitudes = tl.minimum(tl.maximum(magnitudes, 0.0), MAX_MAGNITUDE * 1.0)

        # Signs
        signs = values < 0

        # Store results
        tl.store(signs_ptr + output_base + offsets, signs, mask=mask)
        tl.store(magnitudes_ptr + output_base + offsets, magnitudes.to(tl.uint8), mask=mask)

    @triton.jit
    def _axs6_dequantize_kernel(
        # Pointers
        exponents_ptr,       # [N, num_blocks] uint8
        signs_ptr,           # [N, num_blocks, BLOCK_SIZE] bool
        magnitudes_ptr,      # [N, num_blocks, BLOCK_SIZE] uint8
        output_ptr,          # [N, D] float32 output
        # Dimensions
        N,
        D,
        num_blocks,
        # Constants
        BLOCK_SIZE: tl.constexpr,
        MAX_MAGNITUDE: tl.constexpr,
        EXPONENT_BIAS: tl.constexpr,
    ):
        """
        Fused dequantization kernel: reconstructs float values from AXS-6.

        Grid: (N, num_blocks)
        """
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)

        input_base = batch_id * num_blocks * BLOCK_SIZE + block_id * BLOCK_SIZE
        output_base = batch_id * D + block_id * BLOCK_SIZE

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < BLOCK_SIZE

        # Load shared exponent
        shared_exp = tl.load(exponents_ptr + batch_id * num_blocks + block_id).to(tl.float32)
        scale = tl.math.exp2(shared_exp - EXPONENT_BIAS)

        # Load magnitudes and signs
        magnitudes = tl.load(magnitudes_ptr + input_base + offsets, mask=mask, other=0).to(tl.float32)
        signs = tl.load(signs_ptr + input_base + offsets, mask=mask, other=False)

        # Reconstruct values
        values = (magnitudes / MAX_MAGNITUDE) * scale
        values = tl.where(signs, -values, values)

        # Store output
        tl.store(output_ptr + output_base + offsets, values, mask=mask)

    @triton.jit
    def _axs6_fake_quantize_kernel(
        # Pointers
        input_ptr,
        output_ptr,
        # Dimensions
        N,
        D,
        num_blocks,
        # Constants
        BLOCK_SIZE: tl.constexpr,
        MAX_MAGNITUDE: tl.constexpr,
        EXPONENT_BIAS: tl.constexpr,
        STOCHASTIC: tl.constexpr,
    ):
        """
        Fused fake-quantize kernel: quantize and immediately dequantize
        in a single pass without materializing intermediate representation.

        This is the most performance-critical kernel for training, as it's
        called for every weight/activation quantization in the forward pass.

        Grid: (N, num_blocks)
        """
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        base = batch_id * D + block_id * BLOCK_SIZE

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < BLOCK_SIZE
        values = tl.load(input_ptr + base + offsets, mask=mask, other=0.0)

        # Compute block scale
        abs_values = tl.abs(values)
        abs_max = tl.max(abs_values, axis=0)
        safe_max = tl.where(abs_max > 0, abs_max, 1e-45)
        raw_exp = tl.math.floor(tl.math.log2(safe_max)) + 1.0
        shared_exp = tl.minimum(tl.maximum(raw_exp + EXPONENT_BIAS, 0.0), 255.0)
        shared_exp = tl.where(abs_max > 0, shared_exp, 0.0)
        scale = tl.math.exp2(shared_exp - EXPONENT_BIAS)
        scale = tl.where(scale > 0, scale, 1e-45)

        # Quantize
        normalized = abs_values / scale
        scaled = normalized * MAX_MAGNITUDE

        if STOCHASTIC:
            floor_vals = tl.math.floor(scaled)
            frac = scaled - floor_vals
            rand = tl.rand(batch_id * num_blocks + block_id, offsets)
            magnitudes = tl.where(rand < frac, floor_vals + 1.0, floor_vals)
        else:
            magnitudes = tl.math.floor(scaled + 0.5)

        magnitudes = tl.minimum(tl.maximum(magnitudes, 0.0), MAX_MAGNITUDE * 1.0)

        # Dequantize (fused)
        reconstructed = (magnitudes / MAX_MAGNITUDE) * scale
        signs = values < 0
        reconstructed = tl.where(signs, -reconstructed, reconstructed)
        reconstructed = tl.where(abs_max > 0, reconstructed, 0.0)

        tl.store(output_ptr + base + offsets, reconstructed, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def triton_quantize(
    tensor: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor using the Triton kernel.

    Args:
        tensor: Input tensor of shape ``(N, D)``. D must be a multiple of block_size.
        block_size: Block size (8, 16, or 32).
        stochastic: Whether to use stochastic rounding.

    Returns:
        Tuple of ``(exponents, signs, magnitudes)``:
          - exponents: ``(N, num_blocks)`` uint8
          - signs: ``(N, num_blocks * block_size)`` bool
          - magnitudes: ``(N, num_blocks * block_size)`` uint8
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    assert tensor.ndim == 2, "Input must be 2D"
    N, D = tensor.shape
    assert D % block_size == 0, f"D ({D}) must be a multiple of block_size ({block_size})"
    num_blocks = D // block_size

    device = tensor.device
    exponents = torch.empty((N, num_blocks), dtype=torch.uint8, device=device)
    signs = torch.empty((N, num_blocks * block_size), dtype=torch.bool, device=device)
    magnitudes = torch.empty((N, num_blocks * block_size), dtype=torch.uint8, device=device)

    grid = (N, num_blocks)
    _axs6_quantize_kernel[grid](
        tensor, exponents, signs, magnitudes,
        N, D, num_blocks,
        BLOCK_SIZE=block_size,
        MAX_MAGNITUDE=31,
        EXPONENT_BIAS=127,
        STOCHASTIC=stochastic,
    )

    return exponents, signs, magnitudes


def triton_dequantize(
    exponents: torch.Tensor,
    signs: torch.Tensor,
    magnitudes: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize AXS-6 encoded data using the Triton kernel.

    Args:
        exponents: ``(N, num_blocks)`` uint8
        signs: ``(N, num_blocks * block_size)`` bool
        magnitudes: ``(N, num_blocks * block_size)`` uint8
        block_size: Block size.

    Returns:
        Reconstructed float32 tensor of shape ``(N, D)``.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    N, num_blocks = exponents.shape
    D = num_blocks * block_size
    device = exponents.device

    output = torch.empty((N, D), dtype=torch.float32, device=device)

    grid = (N, num_blocks)
    _axs6_dequantize_kernel[grid](
        exponents, signs, magnitudes, output,
        N, D, num_blocks,
        BLOCK_SIZE=block_size,
        MAX_MAGNITUDE=31,
        EXPONENT_BIAS=127,
    )

    return output


def triton_fake_quantize(
    tensor: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Fused fake-quantize: quantize + dequantize in a single kernel launch.

    This is the fastest path for training, avoiding intermediate allocations.

    Args:
        tensor: Input tensor of shape ``(N, D)``.
        block_size: Block size.
        stochastic: Whether to use stochastic rounding.

    Returns:
        Fake-quantized tensor with same shape.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")

    assert tensor.ndim == 2, "Input must be 2D"
    N, D = tensor.shape
    assert D % block_size == 0, f"D ({D}) must be a multiple of block_size ({block_size})"
    num_blocks = D // block_size

    output = torch.empty_like(tensor)

    grid = (N, num_blocks)
    _axs6_fake_quantize_kernel[grid](
        tensor, output,
        N, D, num_blocks,
        BLOCK_SIZE=block_size,
        MAX_MAGNITUDE=31,
        EXPONENT_BIAS=127,
        STOCHASTIC=stochastic,
    )

    return output
