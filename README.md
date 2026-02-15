# AXS-6: Adaptive eXponent Sharing — 6-Bit Training Format

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

> **A novel 6-bit numerical format with 4× the mantissa precision of FP8, 21% less memory, and a custom Triton kernel that's 2× faster than eager PyTorch. Converges with simple STE — no delayed scaling or loss scaling required.**

## TL;DR — How AXS-6 Compares

**Software fake-quantize comparison** on RTX 5070 Ti, MiniGPT (4-layer Transformer), 200 steps.
All formats use STE (straight-through estimator) — no hardware-native acceleration.

| Format | Bits | ms/step | vs FP32 | Final Loss | Perplexity | Converges? |
|--------|------|---------|---------|------------|------------|:----------:|
| **FP32** | 32 | 9.22 | 1.0× | 0.0533 | 1.05 | **Yes** |
| **AXS-6 (Triton)** | **6.31** | **11.70** | **1.27×** | **0.0537** | **1.06** | **Yes** |
| NF4 | 4 | 18.57 | 2.01× | 6.8432 | 937 | No |
| FP4 E2M1 | 4 | 24.74 | 2.68× | 6.8645 | 958 | No |
| FP8 E4M3 (naive) | 8 | 30.01 | 3.26× | 7.1399 | 1261 | No |

> **Note**: The FP8 result uses a naive software emulation (no delayed scaling, no loss scaling). Production FP8 training with proper infrastructure (torchao, transformer-engine) on FP8-capable hardware (H100/B200) **does converge** and is faster than AXS-6. See [Honest Comparison](#honest-comparison-axs-6-vs-hardware-fp8) below.

**AXS-6's advantage**: it converges with simple STE — no delayed scaling, no amax history, no mixed E4M3/E5M2 — making it the most robust sub-8-bit format for software-level quantization.

## The Problem

Modern AI training uses FP8 (8-bit floating point) as the frontier of low-precision training. FP8 works well with proper infrastructure (delayed scaling, mixed E4M3/E5M2, hardware tensor cores on H100+), but its per-value exponent is redundant — in neural networks, neighboring values within a weight row or channel tend to have similar magnitudes.

This redundancy means FP8 uses 8 bits per value but only gets 3 bits of mantissa precision. And without dedicated infrastructure (delayed scaling, loss scaling, mixed precision), naive FP8/FP4 STE simulations often fail to converge.

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

### Key Advantages

| Metric | FP8 E4M3 | FP4 E2M1 | AXS-6 (B=32) | AXS-6 vs FP8 |
|--------|----------|----------|---------------|---------------|
| Bits per value | 8.00 | 4.00 | 6.31 | **21% smaller** |
| Mantissa bits | 3 | 1 | 5 | **4× more precise** |
| Dynamic range | ±448 | ±6.0 | ±3.4e38 | **Vastly wider** |
| Quantisation levels | 16/scale | 4/scale | 63/scale | **~4× finer** |
| Training convergence | No* | No* | **Yes** | — |

*\*Using naive software STE without delayed scaling or loss scaling. FP8 converges with proper infrastructure (torchao/transformer-engine) on H100/B200. AXS-6 converges even with simple STE.*

## Core Innovation: Fused NF5 Warp Table + Triton Kernel

AXS-6 uses a **NormalFloat-5** quantization grid — 32 non-uniform levels placed at the quantiles of a half-normal distribution, optimal for the Gaussian-distributed weights found in neural networks.

### Fused LUT1024

A precomputed 1024-entry lookup table (4 KB) that maps any normalised `[0, 1]` value directly to its NF5 reconstruction value in **O(1)**. This replaces the traditional encode → intermediate-tensor → decode pipeline with a single gather operation that fits entirely in GPU L1 cache.

```
Traditional quantisation:
  tensor → abs/sign → normalise → encode → intermediate → decode → denormalise → output
  (multiple kernel launches, intermediate allocations)

Fused NF5 (AXS-6):
  tensor → abs/sign → normalise → LUT[x * 1023] → denormalise → output
  (single pass, no intermediates, 4 KB LUT in L1 cache)
```

### Custom Triton Kernel

A hand-written Triton kernel fuses the entire NF5 fake-quantize pipeline into a single GPU pass using **2-D tile vectorisation** — each Triton program processes multiple quantisation blocks simultaneously.

```
Triton kernel strategy:
  ┌────────────────────────────────────────────────┐
  │  Program 0    │  Program 1    │  Program 2 ... │
  │  ┌──────────┐ │  ┌──────────┐ │                │
  │  │ Block 0  │ │  │ Block 32 │ │                │
  │  │ Block 1  │ │  │ Block 33 │ │   ...          │
  │  │   ...    │ │  │   ...    │ │                │
  │  │ Block 31 │ │  │ Block 63 │ │                │
  │  └──────────┘ │  └──────────┘ │                │
  └────────────────────────────────────────────────┘
  Each program: 2-D tile of (N_BLOCKS × BLOCK_SIZE) elements
  → sign, abs, amax, scale, normalise, LUT gather, denorm
  All in one pass, zero intermediate allocations
```

### Fake-Quantize Latency

| Size | Triton | torch.compile | Eager PyTorch | Triton vs Eager |
|------|--------|---------------|---------------|:---------------:|
| 256×256 | 0.024 ms | 0.114 ms | 1.803 ms | **76×** |
| 1024×1024 | 0.019 ms | 0.220 ms | 0.240 ms | **13×** |
| 4096×4096 | 0.393 ms | 7.288 ms | 6.778 ms | **17×** |
| 8192×4096 | 0.798 ms | 11.566 ms | 10.971 ms | **14×** |

## Installation

```bash
# Core (PyTorch + NumPy)
pip install -e .

# With Triton GPU kernels (recommended)
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

### Hardware Backend Selection

The backend is auto-detected (Triton → compiled → eager), but can be overridden:

```python
from axs.unified import get_backend, set_backend, backend_info

# Check what's active
print(get_backend())   # BackendType.TRITON (on CUDA with Triton)
print(backend_info())  # Full diagnostic info

# Force a specific backend
set_backend("compiled")  # torch.compile path
set_backend("triton")    # Custom Triton kernel (fastest)
set_backend("eager")     # Pure PyTorch (always works)

# Or via environment variable
# AXS6_BACKEND=triton python train.py
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

### Custom Triton Kernel
Hand-written Triton kernel with 2-D tile vectorisation for 13–76× faster fake-quantize than eager PyTorch. Auto-detected and used when Triton is available. Includes stochastic rounding support via `tl.rand()` dither.

### Hardware Backend Dispatch
Automatic selection of the fastest available backend:
- **Triton** — Custom kernel, ~15× faster (requires Triton ≥ 3.0)
- **Compiled** — `torch.compile`, ~2× faster (requires PyTorch 2.1+)
- **INT8** — Tensor core matmul for large GEMMs (Turing+ GPUs)
- **Eager** — Pure PyTorch fallback

### Skip-first-N Warmup
A binary flag that bypasses quantisation entirely for the first N training steps. Unlike precision annealing (which interpolates every step), this has **zero runtime overhead** during warmup — it simply returns the input tensor unchanged.

### Stochastic Dithering
Adds uniform noise (±0.5 LUT steps) before the LUT lookup to achieve the stochastic-rounding property needed for unbiased gradient quantisation, at negligible computational cost.

### Power-of-2 Block Scaling
Block scales are constrained to powers of 2 (`2^(floor(log2(amax)) + 1)`), enabling efficient hardware implementation and stable training dynamics.

### Drop-in Modules
- `AXSLinearUnified` — replaces `nn.Linear`
- `AXSLayerNormUnified` — replaces `nn.LayerNorm` (no output quantisation for stability)
- `AXSEmbeddingUnified` — replaces `nn.Embedding` (lazy quantisation: only accessed rows)
- `AXSMultiheadAttentionUnified` — multi-head attention with quantised projections

## Benchmark Results

All benchmarks on NVIDIA RTX 5070 Ti (16 GB), PyTorch 2.10, Python 3.14, Triton 3.6.

### AXS-6 vs FP32 vs FP8 vs FP4 — Software Fake-Quantize Convergence

> **Important**: All formats below use **software fake-quantize (STE)**, not hardware-native acceleration. See [Honest Comparison](#honest-comparison-axs-6-vs-hardware-fp8) for context.

MiniGPT (4-layer Transformer, vocab=1000, dim=256), 200 steps, batch 8×64:

| Format | Bits | ms/step | vs FP32 | Final Loss | Perplexity | Converges? |
|--------|------|---------|---------|------------|------------|:----------:|
| FP32 (baseline) | 32 | 9.22 | 1.0× | 0.0533 | 1.05 | **Yes** |
| **AXS-6 (Triton)** | **6.31** | **11.70** | **1.27×** | **0.0537** | **1.06** | **Yes** |
| NF4 | 4 | 18.57 | 2.01× | 6.8432 | 937 | No |
| FP4 E2M1 | 4 | 24.74 | 2.68× | 6.8645 | 958 | No |
| FP8 E4M3 (naive) | 8 | 30.01 | 3.26× | 7.1399 | 1261 | No |

Key observations:
- AXS-6 converges with simple STE — no delayed scaling or loss scaling needed
- AXS-6 loss (0.0537) is within **0.7%** of FP32 loss (0.0533)
- Naive FP8/FP4/NF4 STE diverge — but this is an artifact of missing infrastructure, not a fundamental FP8 limitation

### Backend Comparison — Fake-Quantize Latency

| Size | Triton | torch.compile | Eager | Triton speedup |
|------|--------|---------------|-------|:--------------:|
| 256×256 | 0.024 ms | 0.114 ms | 1.803 ms | 76× |
| 1024×1024 | 0.019 ms | 0.220 ms | 0.240 ms | 13× |
| 4096×4096 | 0.393 ms | 7.288 ms | 6.778 ms | 17× |
| 8192×4096 | 0.798 ms | 11.566 ms | 10.971 ms | 14× |

### Backend Comparison — End-to-End Training

MiniGPT, 50 training steps:

| Backend | ms/step | vs Eager | Final Loss |
|---------|---------|----------|------------|
| **Triton** | **17.43** | **2.0×** | 0.0090 |
| Compiled | 28.55 | 1.2× | 0.0089 |
| Eager | 34.45 | 1.0× | 0.0092 |

### Quantisation Quality

4096×4096 Gaussian tensor:

| Metric | Value |
|--------|-------|
| MSE | 0.00077 |
| SNR | 31.2 dB |
| MSE reduction vs uniform grid | **34%** |

## Project Structure

```
axs/
├── core.py                    # AXS-6 format: AXSBlock, AXSTensor, encode/decode
├── quantize.py                # Rounding strategies (nearest, stochastic, error feedback, GASR)
├── unified/
│   ├── quantize_unified.py    # Fused NF5 warp table (LUT1024), fake-quantise
│   ├── triton_kernels.py      # Custom Triton kernel (2-D tile vectorisation)
│   ├── backend.py             # Backend dispatch (Triton/compiled/INT8/eager)
│   ├── functional_unified.py  # Autograd ops (STE, quantised linear, matmul)
│   ├── modules_unified.py     # Drop-in layers + convert_to_axs_unified()
│   └── training_unified.py    # Training pipeline + Amax EMA
├── nn/                        # Legacy V1 modules (available via v0.1.0 tag)
└── v2/                        # Legacy V2 modules (available via v0.2.0 tag)

benchmarks/
├── benchmark_triton.py            # Triton vs compiled vs eager
├── benchmark_axs6_vs_fp8_fp4.py   # AXS-6 vs FP32/FP8/FP4/NF4
├── benchmark_unified.py           # Speed + quality + training
└── benchmark_backend.py           # Backend acceleration

tests/
├── test_triton_kernels.py     # 42 Triton kernel tests
├── test_backend.py            # 55 backend tests
├── test_unified.py            # 66 unified implementation tests
├── test_core.py               # V1 format tests (55)
└── test_v2.py                 # V2 tests (37)
```

## Running Tests

```bash
# All tests (323 total)
pytest tests/ -v

# Triton kernel tests (42)
pytest tests/test_triton_kernels.py -v

# Unified + backend (121 tests)
pytest tests/test_unified.py tests/test_backend.py -v
```

## Running Benchmarks

```bash
# Triton vs compiled vs eager
python -m benchmarks.benchmark_triton

# AXS-6 vs FP32/FP8/FP4/NF4
python -m benchmarks.benchmark_axs6_vs_fp8_fp4

# Full benchmark (latency + quality + training)
python -m benchmarks.benchmark_unified
```

## How It Works

### Mixed-Precision Training Pipeline

```
                Forward Pass                    Backward Pass
                ──────────                      ─────────────
FP32 Master ──→ Fused NF5   ──→ AXS-6 ──→ FP32 ──→ Fused NF5 ──→ AXS-6
Weights         Fake-Quantize   Matmul     Grad     Fake-Quantize  Grad
                (nearest)       ───────    ──────   (stochastic)   Comm
                (Triton GPU       │                  (Triton GPU      │
                 kernel)          ▼                   kernel)         ▼
                              FP32 Output          FP32 Weight
                                                   Update (AdamW)
```

### Why Block-Shared Exponents Work for Neural Networks

1. **Weight matrices** — Rows/columns have characteristic scales. A single exponent per block of 32 captures this efficiently.

2. **Post-LayerNorm activations** — LayerNorm normalises to zero mean and unit variance, making values within a block naturally similar in magnitude.

3. **Gradients** — Per-layer gradient magnitudes vary slowly across elements, making shared exponents effective.

4. **Sparsity** — ReLU and GeLU create sparsity, which AXS-6 handles efficiently (zero values use no dynamic range).

### Why FP8/FP4 Fail *With Naive STE* Where AXS-6 Succeeds

> **Clarification**: FP8 *does* converge when using proper infrastructure (delayed scaling, mixed E4M3/E5M2, loss scaling) — as implemented in torchao and NVIDIA transformer-engine. The failures above are specifically with naive software STE (no scaling, no loss scaling).

AXS-6 converges with simple STE where naive FP8 and FP4 don't because:

1. **5 mantissa bits vs 3 (FP8) or 1 (FP4)** — 4× finer quantisation reduces gradient noise below the convergence threshold.
2. **Shared exponent amortisation** — The block exponent captures per-block magnitude once, giving each value the full 5-bit mantissa for precision.
3. **NormalFloat grid** — Non-uniform quantisation levels match the statistical distribution of neural network weights, minimising MSE.
4. **Power-of-2 scaling** — Clean binary scaling avoids the rounding cascades that plague FP8's per-value exponent conversion.

### Honest Comparison: AXS-6 vs Hardware FP8

| | AXS-6 | Hardware FP8 (torchao) |
|---|---|---|
| **Matmul precision** | bf16/fp32 (fake-quantize only) | FP8 tensor cores (~2× throughput) |
| **Mantissa bits** | 5 (better SNR) | 3 |
| **Bits per value** | 6.31 (21% less) | 8.00 |
| **Convergence with simple STE** | Yes | No (needs delayed scaling) |
| **Hardware requirement** | Any CUDA GPU | H100/B200+ for native FP8 |
| **Communication bandwidth** | 6.31 bits | 8 bits |
| **Best use case** | Distributed training, non-H100 GPUs | Single-GPU throughput on H100+ |

**Where AXS-6 genuinely wins**:
- Quantisation quality (5-bit mantissa → better SNR → less degradation)
- Communication bandwidth in distributed training (21% less)
- Robustness (converges with minimal infrastructure)
- Availability (works on any CUDA GPU, not just H100+)

**Where hardware FP8 wins**:
- Raw throughput (~2× via FP8 tensor cores)
- VRAM (FP8 matmul uses less memory than AXS-6's bf16 matmul)
- Ecosystem maturity (torchao, transformer-engine, DeepSpeed)

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
