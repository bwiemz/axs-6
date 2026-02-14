"""
AXS-6 Core Format Implementation
=================================

This module implements the fundamental encoding and decoding operations for the
AXS-6 (Adaptive eXponent Sharing, 6-bit) numerical format.

Format summary:
  - Block of B values (default B=32) sharing a single 8-bit exponent
  - Per-value: 1 sign bit + 5 mantissa bits = 6 bits
  - 2-bit block config field for mode selection
  - Effective cost: 6 + 10/B bits per value (6.3125 for B=32)

The format achieves 4× better mantissa precision than FP8 E4M3 while using
~21% fewer bits per value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AXS6_MANTISSA_BITS: int = 5
AXS6_SIGN_BITS: int = 1
AXS6_VALUE_BITS: int = AXS6_SIGN_BITS + AXS6_MANTISSA_BITS  # 6
AXS6_EXPONENT_BITS: int = 8
AXS6_CONFIG_BITS: int = 2
AXS6_HEADER_BITS: int = AXS6_EXPONENT_BITS + AXS6_CONFIG_BITS  # 10
AXS6_MAX_MAGNITUDE: int = (1 << AXS6_MANTISSA_BITS) - 1  # 31
AXS6_EXPONENT_BIAS: int = 127
DEFAULT_BLOCK_SIZE: int = 32
VALID_BLOCK_SIZES: tuple[int, ...] = (8, 16, 32)


class BlockConfig(IntEnum):
    """Block configuration mode (2-bit field)."""

    DENSE = 0b00  # Standard: all values 6-bit sign-magnitude
    SPARSE = 0b01  # Sparse: bitmask for zeros, remaining values 6-bit
    HIGH_PREC = 0b10  # Mixed: half at 7-bit, half at 5-bit
    RESERVED = 0b11  # Reserved for future extensions


# ---------------------------------------------------------------------------
# AXS Block — fundamental unit of the format
# ---------------------------------------------------------------------------


@dataclass
class AXSBlock:
    """
    A single AXS-6 encoded block.

    Attributes:
        shared_exponent: 8-bit biased exponent (bias=127). The block scale
            is ``2^(shared_exponent - 127)``.
        config: 2-bit mode selector (see :class:`BlockConfig`).
        signs: Boolean array of shape ``(B,)`` — True means negative.
        magnitudes: uint8 array of shape ``(B,)`` with values in ``[0, 31]``.
        block_size: Number of values in this block.
    """

    shared_exponent: int
    config: BlockConfig
    signs: np.ndarray  # bool, shape (B,)
    magnitudes: np.ndarray  # uint8, shape (B,)
    block_size: int = DEFAULT_BLOCK_SIZE

    # -- Derived properties --------------------------------------------------

    @property
    def scale(self) -> float:
        """Block scale factor: 2^(shared_exponent - bias)."""
        return math.ldexp(1.0, self.shared_exponent - AXS6_EXPONENT_BIAS)

    @property
    def total_bits(self) -> int:
        """Total storage bits for this block."""
        return AXS6_HEADER_BITS + self.block_size * AXS6_VALUE_BITS

    @property
    def bits_per_value(self) -> float:
        """Effective bits per value including header overhead."""
        return self.total_bits / self.block_size

    # -- Serialization -------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Pack block into a compact byte representation."""
        # Header: exponent (8 bits) + config (2 bits) → 10 bits, padded to 2 bytes
        header = (self.shared_exponent << 2) | int(self.config)
        header_bytes = header.to_bytes(2, byteorder="little")

        # Pack 6-bit values into a byte stream
        bit_buffer = 0
        bit_count = 0
        payload = bytearray()
        for i in range(self.block_size):
            value_6bit = (int(self.signs[i]) << 5) | int(self.magnitudes[i])
            bit_buffer |= value_6bit << bit_count
            bit_count += 6
            while bit_count >= 8:
                payload.append(bit_buffer & 0xFF)
                bit_buffer >>= 8
                bit_count -= 8
        if bit_count > 0:
            payload.append(bit_buffer & 0xFF)

        return header_bytes + bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> AXSBlock:
        """Unpack a block from its byte representation."""
        header = int.from_bytes(data[:2], byteorder="little")
        shared_exponent = header >> 2
        config = BlockConfig(header & 0x03)

        # Unpack 6-bit values
        payload = data[2:]
        bit_buffer = 0
        bit_count = 0
        byte_idx = 0
        signs = np.zeros(block_size, dtype=bool)
        magnitudes = np.zeros(block_size, dtype=np.uint8)

        for i in range(block_size):
            while bit_count < 6:
                bit_buffer |= payload[byte_idx] << bit_count
                byte_idx += 1
                bit_count += 8
            value_6bit = bit_buffer & 0x3F
            bit_buffer >>= 6
            bit_count -= 6
            signs[i] = bool((value_6bit >> 5) & 1)
            magnitudes[i] = value_6bit & 0x1F

        return cls(
            shared_exponent=shared_exponent,
            config=config,
            signs=signs,
            magnitudes=magnitudes,
            block_size=block_size,
        )


# ---------------------------------------------------------------------------
# AXS Tensor — a fully quantized tensor stored in AXS-6 format
# ---------------------------------------------------------------------------


@dataclass
class AXSTensor:
    """
    A tensor stored in AXS-6 format.

    The original tensor is reshaped so that its last dimension is divided into
    blocks of ``block_size``. Each block is independently quantized.

    Attributes:
        shared_exponents: uint8 tensor of shape ``(*batch_dims, num_blocks)``
        signs: bool tensor of shape ``(*batch_dims, num_blocks, block_size)``
        magnitudes: uint8 tensor of shape ``(*batch_dims, num_blocks, block_size)``
        block_size: Number of elements per block.
        original_shape: Shape of the tensor before quantization.
        num_blocks: Number of blocks along the quantization axis.
    """

    shared_exponents: torch.Tensor  # uint8
    signs: torch.Tensor  # bool
    magnitudes: torch.Tensor  # uint8
    block_size: int
    original_shape: torch.Size
    num_blocks: int

    @property
    def effective_bits_per_value(self) -> float:
        """Average bits per scalar value."""
        return AXS6_VALUE_BITS + AXS6_HEADER_BITS / self.block_size

    @property
    def compression_ratio_vs_fp32(self) -> float:
        """Memory compression ratio compared to FP32."""
        return 32.0 / self.effective_bits_per_value

    @property
    def compression_ratio_vs_fp8(self) -> float:
        """Memory compression ratio compared to FP8."""
        return 8.0 / self.effective_bits_per_value

    @property
    def memory_bytes(self) -> int:
        """Approximate byte count of the quantized representation."""
        numel = 1
        for s in self.original_shape:
            numel *= s
        total_bits = numel * AXS6_VALUE_BITS + (numel // self.block_size) * AXS6_HEADER_BITS
        return math.ceil(total_bits / 8)


# ---------------------------------------------------------------------------
# Encoding (float → AXS-6)
# ---------------------------------------------------------------------------


def axs_encode_block(
    values: np.ndarray,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> AXSBlock:
    """
    Encode a 1-D numpy array into a single AXS-6 block.

    Args:
        values: 1-D float array of length B (must be a valid block size).
        rounding: ``"nearest"`` for deterministic rounding, ``"stochastic"``
            for unbiased stochastic rounding.

    Returns:
        An :class:`AXSBlock` containing the encoded representation.
    """
    block_size = len(values)
    assert block_size in VALID_BLOCK_SIZES, f"Block size must be one of {VALID_BLOCK_SIZES}"

    abs_vals = np.abs(values)
    abs_max = float(np.max(abs_vals))

    if abs_max == 0.0:
        return AXSBlock(
            shared_exponent=0,
            config=BlockConfig.DENSE,
            signs=np.zeros(block_size, dtype=bool),
            magnitudes=np.zeros(block_size, dtype=np.uint8),
            block_size=block_size,
        )

    # Compute shared exponent: smallest power-of-2 that covers abs_max
    raw_exp = math.floor(math.log2(abs_max)) + 1
    shared_exponent = max(0, min(255, raw_exp + AXS6_EXPONENT_BIAS))
    scale = math.ldexp(1.0, shared_exponent - AXS6_EXPONENT_BIAS)

    # Normalize to [0, 1] range and quantize to [0, 31]
    normalized = abs_vals / scale  # in [0, 1]
    scaled = normalized * AXS6_MAX_MAGNITUDE  # in [0, 31]

    if rounding == "stochastic":
        floor_vals = np.floor(scaled)
        frac = scaled - floor_vals
        rand = np.random.uniform(0, 1, size=block_size)
        magnitudes = np.where(rand < frac, floor_vals + 1, floor_vals)
    else:
        magnitudes = np.round(scaled)

    magnitudes = np.clip(magnitudes, 0, AXS6_MAX_MAGNITUDE).astype(np.uint8)
    signs = values < 0

    return AXSBlock(
        shared_exponent=shared_exponent,
        config=BlockConfig.DENSE,
        signs=signs,
        magnitudes=magnitudes,
        block_size=block_size,
    )


def axs_decode_block(block: AXSBlock) -> np.ndarray:
    """
    Decode an AXS-6 block back to floating-point values.

    Args:
        block: An encoded :class:`AXSBlock`.

    Returns:
        1-D float64 numpy array of length ``block.block_size``.
    """
    scale = block.scale
    values = (block.magnitudes.astype(np.float64) / AXS6_MAX_MAGNITUDE) * scale
    values[block.signs] *= -1.0
    return values


# ---------------------------------------------------------------------------
# PyTorch Tensor Quantization (float → AXSTensor → float)
# ---------------------------------------------------------------------------


def quantize(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> AXSTensor:
    """
    Quantize a PyTorch tensor to AXS-6 format.

    The tensor's last dimension is padded (if necessary) and divided into blocks.
    Each block is independently quantized with a shared exponent.

    Args:
        tensor: Input tensor of any shape. Last dimension is quantized.
        block_size: Block size (8, 16, or 32).
        rounding: Rounding mode — ``"nearest"`` or ``"stochastic"``.

    Returns:
        An :class:`AXSTensor` containing the quantized representation.
    """
    assert block_size in VALID_BLOCK_SIZES, f"Block size must be one of {VALID_BLOCK_SIZES}"
    original_shape = tensor.shape
    device = tensor.device

    # Flatten all but last dim, then pad last dim to multiple of block_size
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    last_dim = flat.shape[-1]
    pad_amount = (block_size - last_dim % block_size) % block_size
    if pad_amount > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_amount))

    # Reshape into blocks: (batch, num_blocks, block_size)
    num_blocks = flat.shape[-1] // block_size
    blocked = flat.reshape(-1, num_blocks, block_size)

    # Compute per-block statistics
    abs_vals = blocked.abs()
    abs_max = abs_vals.amax(dim=-1)  # (batch, num_blocks)

    # Shared exponent: ceil(log2(abs_max)), biased
    safe_max = abs_max.clamp(min=1e-45)  # avoid log2(0)
    raw_exp = safe_max.log2().floor().to(torch.int32) + 1
    shared_exponents = (raw_exp + AXS6_EXPONENT_BIAS).clamp(0, 255).to(torch.uint8)

    # Handle all-zero blocks
    zero_blocks = abs_max == 0
    shared_exponents[zero_blocks] = 0

    # Scale: 2^(shared_exponent - bias)
    scales = torch.pow(
        2.0, (shared_exponents.float() - AXS6_EXPONENT_BIAS)
    ).unsqueeze(-1)  # (batch, num_blocks, 1)

    # Normalize and quantize
    scales = scales.clamp(min=1e-45)  # prevent div by zero
    normalized = abs_vals / scales  # (batch, num_blocks, block_size), in [0, 1]
    scaled = normalized * AXS6_MAX_MAGNITUDE  # [0, 31]

    if rounding == "stochastic":
        floor_vals = scaled.floor()
        frac = scaled - floor_vals
        rand = torch.rand_like(frac)
        magnitudes = torch.where(rand < frac, floor_vals + 1, floor_vals)
    else:
        magnitudes = scaled.round()

    magnitudes = magnitudes.clamp(0, AXS6_MAX_MAGNITUDE).to(torch.uint8)
    signs = blocked < 0

    return AXSTensor(
        shared_exponents=shared_exponents.to(device),
        signs=signs.to(device),
        magnitudes=magnitudes.to(device),
        block_size=block_size,
        original_shape=original_shape,
        num_blocks=num_blocks,
    )


def dequantize(axs_tensor: AXSTensor) -> torch.Tensor:
    """
    Dequantize an AXSTensor back to a floating-point PyTorch tensor.

    Args:
        axs_tensor: Quantized tensor in AXS-6 format.

    Returns:
        Reconstructed float32 tensor with the original shape.
    """
    # Reconstruct scales: (batch, num_blocks, 1)
    scales = torch.pow(
        2.0, axs_tensor.shared_exponents.float() - AXS6_EXPONENT_BIAS
    ).unsqueeze(-1)

    # Reconstruct values
    values = (axs_tensor.magnitudes.float() / AXS6_MAX_MAGNITUDE) * scales
    values = torch.where(axs_tensor.signs, -values, values)

    # Reshape back to original
    # First flatten blocks: (batch, num_blocks * block_size)
    batch_size = values.shape[0]
    flat = values.reshape(batch_size, -1)

    # Trim padding from last dim
    orig_last_dim = axs_tensor.original_shape[-1]
    flat = flat[:, :orig_last_dim]

    # Restore original shape
    result = flat.reshape(axs_tensor.original_shape)

    return result


# ---------------------------------------------------------------------------
# Utility: compute quantization error statistics
# ---------------------------------------------------------------------------


def quantization_error(
    original: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    rounding: Literal["nearest", "stochastic"] = "nearest",
) -> dict[str, float]:
    """
    Compute quantization error statistics for AXS-6 encoding of a tensor.

    Returns a dict with keys: ``mse``, ``rmse``, ``max_abs_error``,
    ``mean_abs_error``, ``signal_to_noise_db``.
    """
    axs = quantize(original, block_size=block_size, rounding=rounding)
    reconstructed = dequantize(axs)

    error = (original - reconstructed).float()
    mse = error.pow(2).mean().item()
    rmse = math.sqrt(mse)
    max_abs = error.abs().max().item()
    mean_abs = error.abs().mean().item()

    signal_power = original.float().pow(2).mean().item()
    snr_db = 10 * math.log10(signal_power / max(mse, 1e-45))

    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "signal_to_noise_db": snr_db,
    }
