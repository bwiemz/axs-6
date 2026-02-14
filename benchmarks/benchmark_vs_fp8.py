"""
Benchmark: AXS-6 vs FP8 — Pretraining Speed & Quality Comparison
==================================================================

Head-to-head comparison measuring:
1. Per-step wall-clock time (ms/step)
2. Quantization overhead vs FP32 baseline
3. Convergence quality (loss, perplexity)
4. Memory footprint

Since FP8 requires H100 hardware + torch.float8 (not widely available),
we simulate FP8 E4M3 quantization using the same fake-quantize approach
so the comparison is apples-to-apples on the same hardware.

Run: ``python -m benchmarks.benchmark_vs_fp8``
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.nn.modules import convert_to_axs
from axs.nn.functional import fake_quantize


# ---------------------------------------------------------------------------
# Simulated FP8 E4M3 fake-quantize
# ---------------------------------------------------------------------------

def fp8_e4m3_fake_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP8 E4M3 quantization via clamping + rounding.

    FP8 E4M3: 1 sign, 4 exponent, 3 mantissa bits.
    Range: ±448, precision: 8 mantissa levels per binade.
    """
    max_val = 448.0
    # Clamp to representable range
    clamped = tensor.clamp(-max_val, max_val)
    # Compute scale per-tensor (FP8 typically uses per-tensor or per-channel scaling)
    amax = clamped.abs().max().clamp(min=1e-12)
    scale = max_val / amax
    # Scale, round to 8 mantissa levels, unscale
    # With 3 mantissa bits: 2^3 = 8 representable values per power-of-2 interval
    scaled = clamped * scale
    # Round to nearest representable FP8 value
    sign = scaled.sign()
    magnitude = scaled.abs().clamp(min=1e-12)
    exponent = magnitude.log2().floor()
    # 8 mantissa levels per exponent
    mantissa_step = (2.0 ** exponent) / 8.0
    quantized = sign * ((magnitude / mantissa_step).round() * mantissa_step)
    return quantized / scale


class FP8Linear(nn.Module):
    """Drop-in Linear with simulated FP8 quantization (for fair comparison)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Expose weight/bias directly so MHA's out_proj.weight etc. works
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fp8_e4m3_fake_quantize(self.linear.weight)
        x_q = fp8_e4m3_fake_quantize(x)
        return F.linear(x_q, w_q, self.linear.bias)


def convert_to_fp8(model: nn.Module) -> nn.Module:
    """Convert all nn.Linear layers to FP8Linear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            fp8_layer = FP8Linear(module.in_features, module.out_features, module.bias is not None)
            fp8_layer.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                fp8_layer.linear.bias.data.copy_(module.bias.data)
            setattr(model, name, fp8_layer)
        else:
            convert_to_fp8(module)
    return model


# ---------------------------------------------------------------------------
# Simple Transformer for benchmarking
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    vocab_size: int = 256
    context_len: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.0  # 0 for deterministic comparison


class FeedForward(nn.Module):
    def __init__(self, cfg: BenchConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.embed_dim, cfg.ff_dim)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: BenchConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: BenchConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.cfg = cfg

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_training_benchmark(
    model: nn.Module,
    name: str,
    num_steps: int = 100,
    batch_size: int = 32,
    context_len: int = 64,
    vocab_size: int = 256,
    device: torch.device | None = None,
) -> dict:
    """Run a fixed number of training steps and measure timing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Pre-generate random data
    torch.manual_seed(42)
    data_x = torch.randint(0, vocab_size, (num_steps, batch_size, context_len))
    data_y = torch.randint(0, vocab_size, (num_steps, batch_size, context_len))

    # Warmup (5 steps)
    model.train()
    for i in range(min(5, num_steps)):
        x = data_x[i].to(device)
        y = data_y[i].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed run
    step_times = []
    losses = []

    for i in range(num_steps):
        x = data_x[i].to(device)
        y = data_y[i].to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_times.append((t1 - t0) * 1000)  # ms
        losses.append(loss.item())

    avg_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    max_time = max(step_times)
    std_time = (sum((t - avg_time) ** 2 for t in step_times) / len(step_times)) ** 0.5
    final_loss = losses[-1]
    final_ppl = math.exp(min(final_loss, 20))

    # Tokens per second
    tokens_per_step = batch_size * context_len
    tokens_per_sec = tokens_per_step / (avg_time / 1000)

    print(f"\n  [{name}]")
    print(f"    Step time:     {avg_time:.2f} ms/step  (min={min_time:.2f}, max={max_time:.2f}, std={std_time:.2f})")
    print(f"    Throughput:    {tokens_per_sec:,.0f} tokens/sec")
    print(f"    Final loss:    {final_loss:.4f}")
    print(f"    Final PPL:     {final_ppl:.2f}")

    return {
        "name": name,
        "avg_ms": avg_time,
        "min_ms": min_time,
        "std_ms": std_time,
        "tokens_per_sec": tokens_per_sec,
        "final_loss": final_loss,
        "final_ppl": final_ppl,
        "losses": losses,
        "step_times": step_times,
    }


def benchmark_matmul_overhead() -> None:
    """Quick matmul overhead comparison: FP32 vs FP8-simulated vs AXS-6."""
    print("\n" + "=" * 80)
    print("MATMUL OVERHEAD COMPARISON")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sizes = [(512, 512), (1024, 1024), (2048, 2048)]

    for N, K in sizes:
        A = torch.randn(N, K, device=device)
        B = torch.randn(K, N, device=device)

        warmup = 10
        runs = 50

        # FP32
        for _ in range(warmup):
            _ = A @ B
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = A @ B
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fp32_ms = (time.perf_counter() - t0) / runs * 1000

        # FP8 simulated
        for _ in range(warmup):
            _ = fp8_e4m3_fake_quantize(A) @ fp8_e4m3_fake_quantize(B)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = fp8_e4m3_fake_quantize(A) @ fp8_e4m3_fake_quantize(B)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fp8_ms = (time.perf_counter() - t0) / runs * 1000

        # AXS-6
        for _ in range(warmup):
            _ = fake_quantize(A) @ fake_quantize(B)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = fake_quantize(A) @ fake_quantize(B)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        axs_ms = (time.perf_counter() - t0) / runs * 1000

        print(f"\n  Matmul {N}x{K}:")
        print(f"    FP32:       {fp32_ms:>8.3f} ms  (baseline)")
        print(f"    FP8 (sim):  {fp8_ms:>8.3f} ms  ({fp8_ms/fp32_ms:.2f}x FP32)")
        print(f"    AXS-6:      {axs_ms:>8.3f} ms  ({axs_ms/fp32_ms:.2f}x FP32)")
        print(f"    AXS-6 vs FP8: {axs_ms/fp8_ms:.2f}x")


def main() -> None:
    print("=" * 80)
    print("AXS-6 vs FP8: PRETRAINING SPEED & QUALITY COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    cfg = BenchConfig()
    num_steps = 100
    batch_size = 32

    print(f"\nModel: MiniGPT ({cfg.num_layers}L, {cfg.embed_dim}d, {cfg.num_heads}H)")
    print(f"Batch: {batch_size} × {cfg.context_len} tokens = {batch_size * cfg.context_len:,} tokens/step")
    print(f"Steps: {num_steps}")

    results = {}

    # --- FP32 Baseline ---
    print("\n" + "-" * 80)
    print("1) FP32 BASELINE")
    print("-" * 80)
    torch.manual_seed(42)
    model_fp32 = MiniGPT(cfg)
    print(f"  Params: {count_params(model_fp32):,}")
    results["FP32"] = run_training_benchmark(
        model_fp32, "FP32", num_steps=num_steps, batch_size=batch_size,
        context_len=cfg.context_len, device=device,
    )

    # --- FP8 (simulated) ---
    print("\n" + "-" * 80)
    print("2) FP8 E4M3 (simulated via fake-quantize)")
    print("-" * 80)
    torch.manual_seed(42)
    model_fp8 = MiniGPT(cfg)
    model_fp8 = convert_to_fp8(model_fp8)
    print(f"  Params: {count_params(model_fp8):,}")
    results["FP8"] = run_training_benchmark(
        model_fp8, "FP8 E4M3", num_steps=num_steps, batch_size=batch_size,
        context_len=cfg.context_len, device=device,
    )

    # --- AXS-6 ---
    print("\n" + "-" * 80)
    print("3) AXS-6 (B=32, fake-quantize)")
    print("-" * 80)
    torch.manual_seed(42)
    model_axs = MiniGPT(cfg)
    model_axs = convert_to_axs(model_axs, block_size=32)
    print(f"  Params: {count_params(model_axs):,}")
    results["AXS-6"] = run_training_benchmark(
        model_axs, "AXS-6 (B=32)", num_steps=num_steps, batch_size=batch_size,
        context_len=cfg.context_len, device=device,
    )

    # --- Summary Table ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fp32_time = results["FP32"]["avg_ms"]
    fp32_tps = results["FP32"]["tokens_per_sec"]

    print(f"\n  {'Format':<15} {'ms/step':>10} {'vs FP32':>10} {'tok/s':>12} {'Loss':>10} {'PPL':>10}")
    print(f"  {'─' * 15} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10}")

    for key in ["FP32", "FP8", "AXS-6"]:
        r = results[key]
        ratio = r["avg_ms"] / fp32_time
        print(
            f"  {r['name']:<15} {r['avg_ms']:>10.2f} {ratio:>9.2f}x {r['tokens_per_sec']:>12,.0f} "
            f"{r['final_loss']:>10.4f} {r['final_ppl']:>10.2f}"
        )

    # Speed comparison
    fp8_time = results["FP8"]["avg_ms"]
    axs_time = results["AXS-6"]["avg_ms"]

    print(f"\n  AXS-6 vs FP8 (software simulation):")
    print(f"    Step time ratio: {axs_time / fp8_time:.2f}x")
    print(f"    AXS-6 is {'faster' if axs_time < fp8_time else 'slower'} by {abs(axs_time - fp8_time):.2f} ms/step")

    # Quality comparison
    ppl_diff = results["AXS-6"]["final_ppl"] - results["FP8"]["final_ppl"]
    print(f"\n  AXS-6 vs FP8 (quality after {num_steps} steps):")
    print(f"    PPL delta: {ppl_diff:+.2f}")
    print(f"    Loss delta: {results['AXS-6']['final_loss'] - results['FP8']['final_loss']:+.4f}")

    print(f"\n  Memory (theoretical):")
    print(f"    FP8:   8.00 bits/value")
    print(f"    AXS-6: 6.31 bits/value (block_size=32)")
    print(f"    AXS-6 saves ~21.1% weight memory vs FP8")

    print("\n" + "=" * 80)
    print("IMPORTANT CAVEATS")
    print("=" * 80)
    print("""
  1. This benchmark uses SOFTWARE-EMULATED quantization for both formats.
     Neither FP8 nor AXS-6 use hardware-accelerated quantized math here.

  2. FP8 has NATIVE HARDWARE SUPPORT on NVIDIA H100/H200/B100 GPUs.
     With hardware FP8 (via torch.float8 or Transformer Engine), FP8 matmuls
     are ~2x faster than FP16 on H100. AXS-6 does not have hardware support yet.

  3. The speed comparison here measures QUANTIZATION OVERHEAD only —
     how much extra time the fake-quantize step adds to each training step.
     On real hardware, FP8 wins on compute speed; AXS-6 wins on memory bandwidth.

  4. AXS-6's real advantage is in MEMORY-BANDWIDTH-BOUND scenarios:
     - Large model weights that don't fit in fast memory
     - Distributed training (21% less gradient communication)
     - Inference on memory-constrained devices
     - Activation checkpointing with quantized activations
""")

    # Matmul overhead
    benchmark_matmul_overhead()


if __name__ == "__main__":
    main()
