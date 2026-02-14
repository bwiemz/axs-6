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

### Key Advantages

| Metric | FP8 E4M3 | AXS-6 (B=32) | Improvement |
|--------|----------|---------------|-------------|
| Bits per value | 8.00 | 6.31 | **21.1% smaller** |
| Mantissa bits | 3 | 5 | **4× more precise** |
| Dynamic range | ±448 | ±3.4e38 | **Vastly wider** |
| Quantization levels | 16 per scale | 63 per scale | **~4× finer** |

## Two Implementations: V1 (Fast) and V2 (High Quality)

AXS-6 ships with two quantization backends. Both share the same 6-bit block format, but differ in *how* they map values to quantization codes:

| Feature | V1 | V2 |
|---------|----|----|
| Quantization grid | Uniform | **NormalFloat-5** (Gaussian-optimal) |
| Scale computation | Block max | **Percentile clipping** (outlier-robust) |
| Outlier handling | None | **Hadamard rotation** (optional) |
| Activation balancing | None | **SmoothQuant** (per-channel) |
| Training warmup | None | **Precision annealing** (FP32 → quant) |
| Scale tracking | Per-step | **Amax EMA history** |
| Weighted quantization MSE | baseline | **35.8% lower** |
| SNR (Gaussian data) | 29.4 dB | **31.1 dB** (+1.7 dB) |
| Training speed | baseline | ~18% slower |

**When to use which:**
- **V1** — Maximum throughput, simple integration, good enough quality
- **V2** — Best quality, research experiments, production pretraining where every 0.1 PPL counts

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Core library** | Python + NumPy | Reference encoding/decoding, bit-level operations |
| **Training framework** | PyTorch | Autograd integration, mixed-precision training pipeline |
| **GPU kernels** | Triton | High-performance fused quantize/dequantize/matmul kernels |
| **Optimizer** | Custom AdamW | AXS-aware optimizer with optional gradient quantization |

## Installation

```bash
# Core (PyTorch + NumPy)
pip install -e .

# With Triton GPU kernels
pip install -e ".[triton]"

# With benchmarking tools
pip install -e ".[benchmarks]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Basic Quantization (V1)

```python
import torch
from axs import quantize, dequantize

x = torch.randn(128, 256)
x_axs = quantize(x, block_size=32)
x_restored = dequantize(x_axs)

print(f"Compression: {x_axs.compression_ratio_vs_fp32:.1f}× vs FP32")
print(f"Bits per value: {x_axs.effective_bits_per_value:.2f}")
```

### Basic Quantization (V2 — NormalFloat)

```python
import torch
from axs.v2 import quantize_v2, dequantize_v2

x = torch.randn(128, 256)
x_q, scales = quantize_v2(x, block_size=32, percentile=99.5)
x_restored = dequantize_v2(x_q, scales, block_size=32)
```

### Drop-in Layer Replacement

```python
# V1 — fast path
from axs.nn import AXSLinear
layer = AXSLinear(768, 256, block_size=32)

# V2 — higher quality
from axs.v2 import AXSLinearV2
layer = AXSLinearV2(768, 256, block_size=32)

output = layer(input)  # quantized forward, STE backward
```

### Convert an Entire Model

```python
# V1
from axs.nn import convert_to_axs
model = convert_to_axs(model, block_size=32)

# V2 (also upgrades any existing V1 AXS layers)
from axs.v2 import convert_to_axs_v2
model = convert_to_axs_v2(model, block_size=32)
```

### V2 Training Pipeline with Precision Annealing

```python
from axs.v2 import AXSTrainingPipelineV2, convert_to_axs_v2

model = convert_to_axs_v2(model, block_size=32)
pipeline = AXSTrainingPipelineV2(
    model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4),
    warmup_steps=500,       # start FP32, ramp to full quantization
    use_smooth_quant=True,  # per-channel activation/weight balancing
    max_grad_norm=1.0,
)

for batch in dataloader:
    stats = pipeline.train_step(batch, criterion)
    print(f"step {stats['step']}: loss={stats['loss']:.4f}, "
          f"quant_strength={stats['quant_strength']:.2f}")
```

## Quantization Strategies (V1)

```python
from axs.quantize import (
    quantize_nearest,              # Deterministic — forward pass weights
    quantize_stochastic,           # Unbiased — gradient quantization
    quantize_with_error_feedback,  # Drift-correcting — iterative training
    quantize_gasr,                 # Gradient-aware — fastest convergence
)
```

| Mode | Use Case | Property |
|------|----------|----------|
| **Nearest** | Forward weights, inference | Deterministic, lowest per-sample error |
| **Stochastic** | Gradients | Unbiased: E[Q(x)] = x |
| **Error Feedback** | Iterative training | Corrects accumulated drift |
| **GASR** | Training (novel) | Gradient-aware precision allocation |

### GASR: Gradient-Aware Stochastic Rounding (Novel)

Our novel rounding strategy biases rounding probability based on gradient magnitude. Values with large gradients (high sensitivity to loss) are rounded more accurately, while low-gradient values are rounded more stochastically. This allocates precision where it matters most for optimization.

## V2 Advanced Techniques

### NormalFloat-5 Grid

V2 replaces the uniform quantization grid with 32 non-uniform levels placed at the quantiles of a half-normal distribution. Since neural network weights are approximately Gaussian, this grid minimizes expected quantization error.

### Percentile Clipping

Instead of computing block scale from the absolute maximum (which is sensitive to outliers), V2 uses the 99.5th percentile. This clips the top 0.5% of outliers, reducing scale inflation and improving precision for the remaining 99.5% of values.

### Hadamard Rotation (Optional)

A fast Walsh-Hadamard transform spreads outlier energy across all block dimensions before quantization, then inverts after dequantization. This is especially effective for transformer attention weights where a few channels carry disproportionate magnitude.

### SmoothQuant

Per-channel scaling that migrates quantization difficulty from activations to weights by balancing their dynamic ranges. Supports both offline calibration and online EMA modes.

### Precision Annealing

Training starts at full FP32 precision and linearly ramps quantization strength over a configurable warmup period. This lets the model find a good loss basin before quantization noise kicks in, improving final convergence.

## Project Structure

```
axs/
├── __init__.py              # Public API (V1)
├── core.py                  # Format encoding/decoding, AXSBlock, AXSTensor
├── quantize.py              # Rounding strategies (nearest, stochastic, EF, GASR)
├── utils.py                 # Analysis, comparison, diagnostics
├── nn/
│   ├── functional.py        # V1 autograd functions (STE, quantized ops)
│   ├── modules.py           # V1 drop-in layers (Linear, Conv2d, etc.)
│   └── optim.py             # AXS-aware optimizer + training wrapper
├── v2/
│   ├── __init__.py          # V2 public API
│   ├── quantize_v2.py       # NF5 codebook + percentile clipping
│   ├── hadamard.py          # Walsh-Hadamard rotation
│   ├── smooth_quant.py      # SmoothQuant (calibration + online)
│   ├── annealing.py         # Precision annealing + Amax history
│   ├── functional_v2.py     # V2 autograd functions
│   ├── modules_v2.py        # V2 drop-in layers
│   └── training.py          # V2 training pipeline
└── triton_kernels/
    ├── quantize_kernel.py   # Fused quantize/dequantize GPU kernels
    └── matmul_kernel.py     # Fused quantized matrix multiplication

benchmarks/
├── benchmark_precision.py   # Error comparison vs FP8/FP16/BF16
├── benchmark_memory.py      # Memory footprint analysis
├── benchmark_speed.py       # Throughput benchmarks
├── benchmark_vs_fp8.py      # AXS-6 vs FP8 pretraining comparison
└── benchmark_v1_vs_v2.py    # V1 vs V2 A/B quality benchmark

examples/
├── train_mnist.py           # MNIST training comparison
└── train_transformer.py     # GPT-style transformer pretraining

tests/
├── test_core.py             # Format correctness tests
├── test_quantize.py         # Quantization strategy tests
├── test_nn.py               # Neural network module tests
├── test_training.py         # End-to-end training tests
└── test_v2.py               # V2 component tests (37 tests)
```

## Running Tests

```bash
# All tests (92 total)
pytest tests/ -v

# V1 only
pytest tests/ -v --ignore=tests/test_v2.py

# V2 only
pytest tests/test_v2.py -v
```

## Running Benchmarks

```bash
# V1 vs V2 quality comparison
python -m benchmarks.benchmark_v1_vs_v2

# AXS-6 vs FP8 pretraining speed
python -m benchmarks.benchmark_vs_fp8

# Precision comparison across distributions
python -m benchmarks.benchmark_precision

# Memory footprint for different model sizes
python -m benchmarks.benchmark_memory
```

## Running Examples

```bash
# MNIST training comparison (FP32 vs AXS-6)
python -m examples.train_mnist

# Transformer pretraining comparison
python -m examples.train_transformer
```

## Benchmark Results

Tested on NVIDIA RTX 5070 Ti (16 GB), PyTorch 2.10, Python 3.14:

### Quantization Quality (V1 vs V2)

| Distribution | V1 SNR | V2 SNR | Improvement |
|-------------|--------|--------|-------------|
| Normal(0,1) | 29.4 dB | 31.1 dB | **+1.7 dB** |
| Heavy-tailed | 25.7 dB | 28.1 dB | **+2.5 dB** |
| Outlier (1% at 10×) | 24.8 dB | 27.7 dB | **+2.8 dB** |
| Aggregate MSE | baseline | **-35.8%** | |

### Training Speed (MiniGPT, 200 steps)

| Backend | ms/step | Perplexity |
|---------|---------|------------|
| FP32 | 10.5 | 257.24 |
| AXS-6 V1 | 41.5 | 257.27 |
| AXS-6 V2 | 49.2 | 257.36 |
| FP8 (simulated) | 47.9 | 304.22 |

### vs FP8

AXS-6 matches FP32 training quality (PPL 257 vs 257) while FP8 degrades significantly (PPL 304). AXS-6 achieves 21% memory reduction over FP8.

## Format Specification

See [FORMAT_SPEC.md](FORMAT_SPEC.md) for the complete formal specification including:
- Block structure and encoding algorithms
- Stochastic rounding proofs
- Error feedback mechanism
- Mixed-precision training strategy
- Hardware implementation considerations
- Comparison with MXFP and other block formats

## How It Works

### Mixed-Precision Training Pipeline

```
                Forward Pass                    Backward Pass
                ──────────                      ─────────────
FP32 Master ──→ Quantize to ──→ AXS-6 ──→ FP32 ──→ Quantize ──→ AXS-6
Weights         AXS-6          Matmul     Grad      Gradient     Grad
                (nearest)      ───────    ──────    (stochastic)  Comm
                                  │                      │
                                  ▼                      ▼
                              FP32 Output          FP32 Weight
                                                   Update (AdamW)
```

### Why Block-Shared Exponents Work for Neural Networks

1. **Weight matrices** — Rows/columns of trained weight matrices have characteristic scales. A single exponent per block of 32 captures this well.

2. **Post-LayerNorm activations** — LayerNorm normalizes each feature to zero mean and unit variance, making values within a block naturally similar in magnitude.

3. **Gradients** — Per-layer gradient magnitudes vary slowly across elements, making shared exponents effective.

4. **Sparsity** — ReLU and GeLU activations create sparsity, which AXS-6 handles efficiently (zero values use no dynamic range).

## Theoretical Foundation

**Quantization noise bound** for AXS-6 with block size B=32:
- Step size: Δ = 2S/62 (where S is the block scale)
- Quantization noise variance: σ² = Δ²/12
- SNR: 10·log₁₀(σ²_signal/σ²_noise) ≈ 30–35 dB for typical weight distributions

This is **~12 dB better than FP8 E4M3** (which has only 16 levels per scale), translating to roughly **4× less quantization noise**.

## License

[Apache 2.0](LICENSE)
