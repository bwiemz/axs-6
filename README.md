# AXS-6: Adaptive eXponent Sharing — 6-Bit Training Format

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

> **A novel 6-bit numerical format that achieves 21% memory reduction over FP8 with 4× better intra-block precision, designed specifically for efficient deep learning training.**

## The Problem

Modern AI training uses FP8 (8-bit floating point) as the frontier of low-precision training. But FP8's per-value exponent is wasteful — in neural networks, neighboring values (within a weight row, channel, or attention head) tend to have similar magnitudes. Each FP8 value redundantly encodes its own exponent.

## Our Solution: AXS-6

**AXS-6 (Adaptive eXponent Sharing)** factors out the exponent into a single 8-bit shared scale per block of 32 values, freeing all remaining bits for mantissa precision:

```
┌───────────────────────────────────────────────────────┐
│  Standard FP8 E4M3 (per value)                        │
│  [sign:1][exponent:4][mantissa:3] = 8 bits, 3-bit     │
│                                                       │
│  AXS-6 Block (shared across 32 values)                │
│  [shared_exponent:8][config:2]  ← 10 bits, once       │
│  [sign:1][mantissa:5] × 32     ← 6 bits each          │
│                                                       │
│  Effective: 6.31 bits/value, 5-bit precision           │
└───────────────────────────────────────────────────────┘
```

### Key Advantages over FP8

| Metric | FP8 E4M3 | AXS-6 (B=32) | Improvement |
|--------|----------|---------------|-------------|
| Bits per value | 8.00 | 6.31 | **21.1% smaller** |
| Mantissa bits | 3 | 5 | **4× more precise** |
| Dynamic range | ±448 | ±3.4e38 | **Vastly wider** |
| Quantization levels | 16 per scale | 63 per scale | **~4× finer** |

## Core Innovation: Fused NF5 Warp Table

AXS-6 uses a **NormalFloat-5** quantization grid — 32 non-uniform levels placed at the quantiles of a half-normal distribution, optimal for the Gaussian-distributed weights found in neural networks.

The key breakthrough is the **fused NF5 warp table**: a precomputed 1024-entry lookup table (4 KB) that maps any normalised `[0, 1]` value directly to its NF5 reconstruction value in **O(1)**. This replaces the traditional encode → intermediate-tensor → decode pipeline with a single gather operation that fits entirely in GPU L1 cache.

```
Traditional quantisation:
  tensor → abs/sign → normalise → encode → intermediate → decode → denormalise → output
  (multiple kernel launches, intermediate allocations)

Fused NF5 (AXS-6):
  tensor → abs/sign → normalise → LUT[x * 1023] → denormalise → output
  (single pass, no intermediates, 4 KB LUT in L1 cache)
```

### Why It's Fast

The 4 KB LUT fits in L1 cache on any modern GPU. A single `gather` from L1 is faster than the arithmetic it replaces (`round` + `clamp` + `divide`). The entire fake-quantize round-trip happens in one fused function with zero intermediate tensor allocations.

## Installation

```bash
# Core (PyTorch + NumPy)
pip install -e .

# With Triton GPU kernels
pip install -e ".[triton]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Quantize a Tensor

```python
import torch
from axs.unified import fused_fake_quantize, quantize_unified, dequantize_unified

x = torch.randn(128, 256)

# Fast training path (no intermediate allocation)
x_q = fused_fake_quantize(x, block_size=32)

# Serialisation path (for checkpoints / inference)
x_axs = quantize_unified(x, block_size=32)
x_restored = dequantize_unified(x_axs)

print(f"Compression: {x_axs.compression_ratio_vs_fp32:.1f}× vs FP32")
print(f"Bits per value: {x_axs.effective_bits_per_value:.2f}")
```

### Drop-in Layer Replacement

```python
from axs.unified import AXSLinearUnified

# Replace nn.Linear anywhere in your model
layer = AXSLinearUnified(768, 256, block_size=32)
output = layer(input)  # quantised forward, STE backward
```

### Convert an Entire Model

```python
from axs.unified import convert_to_axs_unified

model = convert_to_axs_unified(model, block_size=32)
# All nn.Linear, nn.LayerNorm, nn.Embedding layers are now AXS-6 quantised
```

### Training Pipeline

```python
from axs.unified import AXSTrainingPipelineUnified, convert_to_axs_unified

model = convert_to_axs_unified(model, block_size=32)
pipeline = AXSTrainingPipelineUnified(
    model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4),
    warmup_steps=500,      # first 500 steps in FP32, zero overhead
    max_grad_norm=1.0,
)

for batch in dataloader:
    stats = pipeline.training_step(batch, criterion)
    print(f"step {stats['step']}: loss={stats['loss']:.4f}, "
          f"warmup={stats['warmup']}")
```

## Features

### Fused NF5 Warp Table (LUT1024)
A precomputed 1024-entry lookup table that maps normalised values to NF5 reconstruction values in O(1). Eliminates the encode → intermediate → decode pipeline, achieving 31% faster training than a uniform quantisation grid.

### Skip-first-N Warmup
A binary flag that bypasses quantisation entirely for the first N training steps. Unlike precision annealing (which interpolates every step), this has **zero runtime overhead** during warmup — it simply returns the input tensor unchanged.

### Stochastic Dithering
Adds uniform noise (±0.5 LUT steps) before the LUT lookup to achieve the stochastic-rounding property needed for unbiased gradient quantisation, at negligible computational cost.

### Power-of-2 Block Scaling
Block scales are constrained to powers of 2 (`2^(floor(log2(amax)) + 1)`), enabling efficient hardware implementation and stable training dynamics.

### Optional Amax EMA (Delayed Scaling)
An exponential moving average of per-tensor amax values following the FP8-LM approach. Reuses the previous step's scale instead of computing a fresh amax reduction each forward pass.

### Drop-in Modules
- `AXSLinearUnified` — replaces `nn.Linear`
- `AXSLayerNormUnified` — replaces `nn.LayerNorm` (no output quantisation for stability)
- `AXSEmbeddingUnified` — replaces `nn.Embedding` (lazy quantisation: only accessed rows)
- `AXSMultiheadAttentionUnified` — multi-head attention with quantised projections

## Benchmark Results

Tested on NVIDIA RTX 5070 Ti (16 GB), PyTorch 2.10, Python 3.14:

### Training Speed (MiniGPT, 200 steps)

| Backend | ms/step | vs FP32 | Final Loss |
|---------|---------|---------|------------|
| FP32 (baseline) | 6.83 | 1.0× | 0.0560 |
| **AXS-6** | **22.98** | 3.4× | **0.0578** |

### Quantisation Quality (4096×4096 Gaussian tensor)

| Metric | Value |
|--------|-------|
| MSE | 0.00077 |
| SNR | 31.2 dB |
| MSE reduction vs uniform grid | **34%** |

### Fake-Quantize Latency (4096×4096, GPU)

| Operation | Latency |
|-----------|---------|
| AXS-6 fused fake-quantize | 6.33 ms |

### vs FP8

AXS-6 matches FP32 training quality where FP8 degrades significantly. AXS-6 achieves **21% memory reduction** over FP8 with **4× less quantisation noise** (~12 dB better SNR).

## Project Structure

```
axs/
├── core.py                    # AXS-6 format: AXSBlock, AXSTensor, encode/decode
├── quantize.py                # Rounding strategies (nearest, stochastic, error feedback, GASR)
├── unified/
│   ├── quantize_unified.py    # Fused NF5 warp table (LUT1024), fake-quantise
│   ├── functional_unified.py  # Autograd ops (STE, quantised linear, matmul)
│   ├── modules_unified.py     # Drop-in layers + convert_to_axs_unified()
│   └── training_unified.py    # Training pipeline + Amax EMA
├── nn/                        # Legacy V1 modules (available via v0.1.0 tag)
└── v2/                        # Legacy V2 modules (available via v0.2.0 tag)

benchmarks/
├── benchmark_unified.py       # Speed + quality + training benchmark
└── ...

tests/
├── test_unified.py            # 66 tests for unified implementation
├── test_core.py               # Format correctness tests
└── ...
```

## Running Tests

```bash
# All tests (158 total)
pytest tests/ -v

# Unified only (66 tests)
pytest tests/test_unified.py -v
```

## Running Benchmarks

```bash
# Full benchmark (latency + quality + training)
python -m benchmarks.benchmark_unified

# AXS-6 vs FP8 pretraining
python -m benchmarks.benchmark_vs_fp8
```

## How It Works

### Mixed-Precision Training Pipeline

```
                Forward Pass                    Backward Pass
                ──────────                      ─────────────
FP32 Master ──→ Fused NF5   ──→ AXS-6 ──→ FP32 ──→ Fused NF5 ──→ AXS-6
Weights         Fake-Quantize   Matmul     Grad     Fake-Quantize  Grad
                (nearest)       ───────    ──────   (stochastic)   Comm
                                  │                      │
                                  ▼                      ▼
                              FP32 Output          FP32 Weight
                                                   Update (AdamW)
```

### Why Block-Shared Exponents Work for Neural Networks

1. **Weight matrices** — Rows/columns have characteristic scales. A single exponent per block of 32 captures this efficiently.

2. **Post-LayerNorm activations** — LayerNorm normalises to zero mean and unit variance, making values within a block naturally similar in magnitude.

3. **Gradients** — Per-layer gradient magnitudes vary slowly across elements, making shared exponents effective.

4. **Sparsity** — ReLU and GeLU create sparsity, which AXS-6 handles efficiently (zero values use no dynamic range).

## Theoretical Foundation

**Quantisation noise bound** for AXS-6 with block size B=32:
- Step size: Δ = 2S/62 (where S is the block scale)
- Quantisation noise variance: σ² = Δ²/12
- SNR: 10·log₁₀(σ²_signal/σ²_noise) ≈ 30–35 dB for typical weight distributions

This is **~12 dB better than FP8 E4M3** (which has only 16 levels per scale), translating to roughly **4× less quantisation noise**.

## Legacy Versions

Previous iterations of the quantiser are available as git tags:

- **`v0.1.0`** — V1: Uniform quantisation grid, simple max scaling
- **`v0.2.0`** — V2: NormalFloat-5 grid, percentile clipping, Hadamard rotation, SmoothQuant, precision annealing

```bash
git checkout v0.1.0  # V1
git checkout v0.2.0  # V2
```

The V1 and V2 source code remains in `axs/nn/` and `axs/v2/` for backwards compatibility, but the unified backend (`axs/unified/`) is recommended for all new work.

## Format Specification

See [FORMAT_SPEC.md](FORMAT_SPEC.md) for the complete formal specification.

## License

[Apache 2.0](LICENSE)
