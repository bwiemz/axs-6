"""
AXS-6 Unified vs V1 vs V2 Benchmark
=====================================

Measures:
  1. Fake-quantize latency (the hot path during training)
  2. MSE (quantisation quality)
  3. SNR (signal-to-noise ratio)
  4. MiniGPT training: 200 steps, loss and wall-clock time
"""

from __future__ import annotations

import time
import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Fake-quantize micro-benchmark
# ---------------------------------------------------------------------------

def bench_fake_quantize():
    print("=" * 70)
    print("FAKE-QUANTIZE LATENCY  (4096×4096 tensor, GPU)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4096, 4096, device=device)

    # V1: quantize → dequantize
    from axs.core import quantize, dequantize
    # V2: quantize_v2 → dequantize_v2
    from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
    # Unified: fused_fake_quantize
    from axs.unified.quantize_unified import fused_fake_quantize

    results = {}

    # --- V1 ---
    for _ in range(5):  # warmup
        _ = dequantize(quantize(x))
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = dequantize(quantize(x))
    if device == "cuda":
        torch.cuda.synchronize()
    v1_ms = (time.perf_counter() - t0) / 20 * 1000
    results["V1"] = v1_ms

    # --- V2 (NF5, no percentile for fair comparison) ---
    for _ in range(5):
        _ = dequantize_v2(quantize_v2(x, use_nf5=True, clip_percentile=None), use_nf5=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = dequantize_v2(quantize_v2(x, use_nf5=True, clip_percentile=None), use_nf5=True)
    if device == "cuda":
        torch.cuda.synchronize()
    v2_ms = (time.perf_counter() - t0) / 20 * 1000
    results["V2"] = v2_ms

    # --- Unified ---
    for _ in range(5):
        _ = fused_fake_quantize(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = fused_fake_quantize(x)
    if device == "cuda":
        torch.cuda.synchronize()
    uni_ms = (time.perf_counter() - t0) / 20 * 1000
    results["Unified"] = uni_ms

    for name, ms in results.items():
        print(f"  {name:10s}: {ms:8.3f} ms")

    fastest = min(results, key=results.get)
    print(f"\n  Fastest: {fastest}")
    print(f"  Unified speedup vs V1: {v1_ms / uni_ms:.2f}×")
    print(f"  Unified speedup vs V2: {v2_ms / uni_ms:.2f}×")
    return results


# ---------------------------------------------------------------------------
# 2. Quality comparison
# ---------------------------------------------------------------------------

def bench_quality():
    print("\n" + "=" * 70)
    print("QUANTISATION QUALITY  (4096×4096 Gaussian tensor)")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(4096, 4096)

    from axs.core import quantize, dequantize
    from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
    from axs.unified.quantize_unified import fused_fake_quantize

    # V1
    v1_out = dequantize(quantize(x))
    v1_mse = (x - v1_out).pow(2).mean().item()

    # V2
    v2_out = dequantize_v2(quantize_v2(x, use_nf5=True, clip_percentile=None), use_nf5=True)
    v2_mse = (x - v2_out).pow(2).mean().item()

    # Unified
    uni_out = fused_fake_quantize(x)
    uni_mse = (x - uni_out).pow(2).mean().item()

    signal_power = x.pow(2).mean().item()
    for name, mse in [("V1", v1_mse), ("V2", v2_mse), ("Unified", uni_mse)]:
        snr = 10 * math.log10(signal_power / max(mse, 1e-45))
        print(f"  {name:10s}: MSE={mse:.8f}  SNR={snr:.2f} dB")

    print(f"\n  Unified MSE reduction vs V1: {(1 - uni_mse/v1_mse)*100:.1f}%")
    print(f"  Unified MSE vs V2: {(uni_mse/v2_mse - 1)*100:+.2f}%")


# ---------------------------------------------------------------------------
# 3. MiniGPT training benchmark
# ---------------------------------------------------------------------------

class MiniGPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab: int = 1000, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, seq_len: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.blocks = nn.Sequential(*[MiniGPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx) + self.pos_embed[:, :idx.shape[1]]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def bench_training(steps: int = 200):
    print("\n" + "=" * 70)
    print(f"MINIGPT TRAINING  ({steps} steps, vocab=1000, d=128, 2 layers)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from axs.nn.modules import convert_to_axs
    from axs.v2.modules_v2 import convert_to_axs_v2
    from axs.unified.modules_unified import convert_to_axs_unified

    configs = [
        ("FP32 (baseline)", lambda m: m),
        ("V1", lambda m: convert_to_axs(m)),
        ("V2", lambda m: convert_to_axs_v2(m, use_hadamard=False)),
        ("Unified", lambda m: convert_to_axs_unified(m)),
    ]

    for name, convert_fn in configs:
        torch.manual_seed(0)
        base = MiniGPT().to(device)
        model = convert_fn(base).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(3):
            idx = torch.randint(0, 1000, (8, 32), device=device)
            logits = model(idx)
            loss = nn.functional.cross_entropy(logits.view(-1, 1000), idx.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        losses = []
        for step in range(steps):
            idx = torch.randint(0, 1000, (8, 32), device=device)
            logits = model(idx)
            loss = nn.functional.cross_entropy(logits.view(-1, 1000), idx.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0 or step == steps - 1:
                losses.append(loss.item())

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms_per_step = elapsed / steps * 1000

        print(f"\n  {name}:")
        print(f"    Time: {ms_per_step:.2f} ms/step  ({elapsed:.1f}s total)")
        print(f"    Loss: {' → '.join(f'{l:.2f}' for l in losses)}")
        print(f"    Final loss: {losses[-1]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench_fake_quantize()
    bench_quality()
    bench_training()
