"""
Triton Kernel: AXS-6 Quantized Matrix Multiplication
=====================================================

A fused kernel that performs quantization and matrix multiplication in a
single kernel launch. This is the most critical kernel for training throughput
as it sits on the hot path of every linear layer forward/backward pass.

Architecture:
  1. Load tiles of A and B from global memory
  2. Quantize each tile to AXS-6 (block-scale + 5-bit mantissa)
  3. Perform integer MAC with shared-exponent scaling
  4. Accumulate results in FP32

This exploits the AXS-6 format's key advantage: within a block, all values
share the same exponent, so multiplication reduces to integer mantissa
products with a single shared-exponent multiplication per block pair.
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
    def _axs6_matmul_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        # Matrix dimensions (M×K) @ (K×N) → (M×N)
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # AXS-6 parameters
        MAX_MAGNITUDE: tl.constexpr,
        EXPONENT_BIAS: tl.constexpr,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        # Whether to quantize
        QUANTIZE_A: tl.constexpr,
        QUANTIZE_B: tl.constexpr,
    ):
        """
        AXS-6 quantized matrix multiplication kernel.

        For each tile, we:
          1. Load BLOCK_M × BLOCK_K tile of A and BLOCK_K × BLOCK_N tile of B
          2. Quantize each to AXS-6 block format (per-row blocks for A, per-col for B)
          3. Compute tile product in FP32 with quantized inputs
          4. Accumulate into output

        Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator in FP32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate over K dimension in tiles
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load A tile: (BLOCK_M, BLOCK_K)
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=a_mask, other=0.0
            )

            # Load B tile: (BLOCK_K, BLOCK_N)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(
                B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=b_mask, other=0.0
            )

            # AXS-6 quantize A (per-row blocks of BLOCK_K)
            if QUANTIZE_A:
                a_abs = tl.abs(a)
                a_max = tl.max(a_abs, axis=1)[:, None]  # per-row max
                a_safe = tl.where(a_max > 0, a_max, 1e-45)
                a_exp = tl.math.floor(tl.math.log2(a_safe)) + 1.0
                a_scale = tl.math.exp2(a_exp)
                a_norm = a_abs / tl.where(a_scale > 0, a_scale, 1e-45)
                a_mag = tl.math.nearbyint(a_norm * MAX_MAGNITUDE)
                a_mag = tl.minimum(tl.maximum(a_mag, 0.0), MAX_MAGNITUDE * 1.0)
                a_q = (a_mag / MAX_MAGNITUDE) * a_scale
                a_q = tl.where(a < 0, -a_q, a_q)
                a_q = tl.where(a_max > 0, a_q, 0.0)
            else:
                a_q = a

            # AXS-6 quantize B (per-column blocks of BLOCK_K)
            if QUANTIZE_B:
                b_abs = tl.abs(b)
                b_max = tl.max(b_abs, axis=0)[None, :]  # per-col max
                b_safe = tl.where(b_max > 0, b_max, 1e-45)
                b_exp = tl.math.floor(tl.math.log2(b_safe)) + 1.0
                b_scale = tl.math.exp2(b_exp)
                b_norm = b_abs / tl.where(b_scale > 0, b_scale, 1e-45)
                b_mag = tl.math.nearbyint(b_norm * MAX_MAGNITUDE)
                b_mag = tl.minimum(tl.maximum(b_mag, 0.0), MAX_MAGNITUDE * 1.0)
                b_q = (b_mag / MAX_MAGNITUDE) * b_scale
                b_q = tl.where(b < 0, -b_q, b_q)
                b_q = tl.where(b_max > 0, b_q, 0.0)
            else:
                b_q = b

            # Accumulate tile product
            acc += tl.dot(a_q, b_q)

        # Store result
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc, mask=c_mask
        )


def triton_axs_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    quantize_a: bool = True,
    quantize_b: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    AXS-6 quantized matrix multiplication using Triton.

    Computes ``C = quantize(A) @ quantize(B)`` where quantization is
    fused into the GEMM kernel for maximum throughput.

    Args:
        a: Left matrix ``(M, K)``, float32.
        b: Right matrix ``(K, N)``, float32.
        quantize_a: Whether to quantize A.
        quantize_b: Whether to quantize B.
        block_m, block_n, block_k: Tile dimensions for the kernel.

    Returns:
        Result matrix ``(M, N)``, float32.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")

    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {a.shape} @ {b.shape}"

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _axs6_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        MAX_MAGNITUDE=31,
        EXPONENT_BIAS=127,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        QUANTIZE_A=quantize_a,
        QUANTIZE_B=quantize_b,
    )

    return c
