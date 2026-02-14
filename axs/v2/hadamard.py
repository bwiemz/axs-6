"""
Hadamard Rotation for Outlier Spreading
========================================

The core insight from QuIP# (Chee et al., 2023): applying a random orthogonal
transform before quantization "spreads" outlier magnitudes across all elements
in a block, making the shared exponent less sensitive to any single value.

For AXS-6 with block_size=32, we use 32×32 Hadamard matrices, which are:
  - Orthogonal (preserves vector norms → same signal energy)
  - Composed only of +1/-1 entries (no multiplications needed, just add/subtract)
  - O(B log B) to apply via fast Walsh-Hadamard transform
  - Perfectly invertible (H^T = H / B)

The effect on quantization:
  - Before rotation: one outlier at 100, rest at ~1 → shared_exp set by 100,
    all other values use <1% of the code range → ~1 effective bit of precision
  - After rotation: outlier is spread to ~100/√32 ≈ 17.7 per element,
    other values grow to ~1*√1 ≈ 1 → shared_exp set by ~17.7,
    all values use ~6-100% of code range → ~4-5 effective bits

Expected improvement: 25-40% MSE reduction on layers with outlier channels.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Hadamard Matrix Construction
# ---------------------------------------------------------------------------

def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Construct a normalized Hadamard matrix of size n×n.

    Uses the Sylvester construction: H(1) = [1], H(2n) = [[H(n), H(n)], [H(n), -H(n)]].
    The resulting matrix satisfies H @ H^T = I (orthogonal).

    Args:
        n: Size of the matrix. Must be a power of 2.
        dtype: Desired dtype.

    Returns:
        Normalized Hadamard matrix of shape (n, n) where H @ H^T = I.
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"

    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    # Normalize so H @ H^T = I
    H = H / math.sqrt(n)
    return H


# Cache Hadamard matrices for common block sizes
_hadamard_cache: dict[tuple[int, str], torch.Tensor] = {}


def get_hadamard(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Get a cached Hadamard matrix for the given size and device."""
    key = (n, str(device))
    if key not in _hadamard_cache:
        _hadamard_cache[key] = hadamard_matrix(n).to(device)
    return _hadamard_cache[key]


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (O(B log B) instead of O(B²))
# ---------------------------------------------------------------------------

def fast_walsh_hadamard(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply the Walsh-Hadamard transform along the last dimension.

    This is O(B log B) vs O(B²) for explicit matrix multiplication,
    and requires no multiplications — only additions and subtractions.

    Args:
        x: Input tensor of shape (..., B) where B is a power of 2.
        normalize: If True, normalize by 1/√B for orthogonality.

    Returns:
        Transformed tensor of same shape.
    """
    B = x.shape[-1]
    assert B > 0 and (B & (B - 1)) == 0, f"Last dim must be power of 2, got {B}"

    result = x.clone()
    h = 1
    while h < B:
        # Process pairs of elements at distance h
        for i in range(0, B, h * 2):
            a = result[..., i:i + h].clone()
            b = result[..., i + h:i + 2 * h].clone()
            result[..., i:i + h] = a + b
            result[..., i + h:i + 2 * h] = a - b
        h *= 2

    if normalize:
        result = result / math.sqrt(B)

    return result


# ---------------------------------------------------------------------------
# Block-level Rotation for Quantization
# ---------------------------------------------------------------------------

def rotate_blocks(
    blocked: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Apply Hadamard rotation to each block independently.

    For a block of shape (..., block_size), applies:
      - Forward: y = H @ x  (spread outliers)
      - Inverse: x = H^T @ y = H @ y  (Hadamard is self-inverse up to scale)

    Since the Hadamard matrix is symmetric and orthogonal (H = H^T, H @ H = I),
    the forward and inverse transforms are identical.

    Args:
        blocked: Tensor of shape (..., num_blocks, block_size).
        inverse: Whether to apply inverse rotation (same as forward for Hadamard).

    Returns:
        Rotated tensor of same shape.
    """
    block_size = blocked.shape[-1]
    # Hadamard is its own inverse, so forward == inverse
    return fast_walsh_hadamard(blocked, normalize=True)


# ---------------------------------------------------------------------------
# Random Sign Flipping for Additional Decorrelation
# ---------------------------------------------------------------------------

_random_signs_cache: dict[tuple[int, str], torch.Tensor] = {}


def get_random_signs(
    block_size: int,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """
    Get a fixed random sign vector for a given block size.

    This is applied before the Hadamard transform for additional decorrelation
    (following QuIP#). The signs are fixed per block_size so the transform
    is deterministic and reproducible.

    Args:
        block_size: Number of elements per block.
        device: Target device.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of +1/-1 values with shape (block_size,).
    """
    key = (block_size, str(device))
    if key not in _random_signs_cache:
        gen = torch.Generator()
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (block_size,), generator=gen).float() * 2 - 1
        _random_signs_cache[key] = signs.to(device)
    return _random_signs_cache[key]


def apply_hadamard_rotation(
    blocked: torch.Tensor,
    use_random_signs: bool = True,
) -> torch.Tensor:
    """
    Full Hadamard rotation pipeline: random sign flip + WHT.

    This is the recommended pre-quantization transform.

    Args:
        blocked: Tensor of shape (..., num_blocks, block_size).
        use_random_signs: Whether to apply random sign flipping.

    Returns:
        Rotated tensor ready for quantization.
    """
    block_size = blocked.shape[-1]
    device = blocked.device

    if use_random_signs:
        signs = get_random_signs(block_size, device)
        blocked = blocked * signs

    return fast_walsh_hadamard(blocked, normalize=True)


def invert_hadamard_rotation(
    blocked: torch.Tensor,
    use_random_signs: bool = True,
) -> torch.Tensor:
    """
    Inverse of apply_hadamard_rotation.

    Since Hadamard is self-inverse (up to normalization, which we handle),
    this applies WHT then undoes the sign flip.
    """
    block_size = blocked.shape[-1]
    device = blocked.device

    result = fast_walsh_hadamard(blocked, normalize=True)

    if use_random_signs:
        signs = get_random_signs(block_size, device)
        result = result * signs  # sign flip is its own inverse

    return result
