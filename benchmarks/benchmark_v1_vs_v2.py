"""
A/B Benchmark: AXS-6 v1 vs v2 vs FP8 vs FP32
=================================================

Comprehensive head-to-head comparison measuring:
  1. Quantization error (MSE, SNR) on random Gaussian tensors
  2. Per-step training time on MiniGPT
  3. Convergence quality (loss, perplexity) over 200 steps
  4. Memory footprint estimation
  5. Per-layer error analysis

Run: ``python -m benchmarks.benchmark_v1_vs_v2``
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.core import quantize, dequantize, quantization_error, DEFAULT_BLOCK_SIZE
from axs.nn.modules import convert_to_axs
from axs.v2.quantize_v2 import quantize_v2, dequantize_v2
from axs.v2.modules_v2 import convert_to_axs_v2


# ---------------------------------------------------------------------------
# FP8 simulation (reuse from existing benchmark)
# ---------------------------------------------------------------------------


def fp8_e4m3_fake_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 E4M3: 1 sign, 4 exponent, 3 mantissa."""
    max_val = 448.0
    clamped = tensor.clamp(-max_val, max_val)
    amax = clamped.abs().max().clamp(min=1e-12)
    scale = max_val / amax
    scaled = clamped * scale
    sign = scaled.sign()
    magnitude = scaled.abs().clamp(min=1e-12)
    exponent = magnitude.log2().floor()
    mantissa_step = (2.0 ** exponent) / 8.0
    quantized = sign * ((magnitude / mantissa_step).round() * mantissa_step)
    return quantized / scale


# ---------------------------------------------------------------------------
# MiniGPT model
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    vocab_size: int = 256
    context_len: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.0


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
        self.attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
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
# Test 1: Quantization Error Analysis
# ---------------------------------------------------------------------------


def test_quantization_error() -> None:
    """Compare quantization error across methods on diverse distributions."""
    print("=" * 70)
    print("TEST 1: QUANTIZATION ERROR ANALYSIS")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distributions = {
        "Normal(0, 1)": lambda: torch.randn(1024, 128, device=device),
        "Normal(0, 0.01)": lambda: torch.randn(1024, 128, device=device) * 0.01,
        "Uniform(-1, 1)": lambda: torch.rand(1024, 128, device=device) * 2 - 1,
        "Heavy-tailed (t-dist)": lambda: torch.distributions.StudentT(df=3).sample(
            (1024, 128)
        ).to(device),
        "Sparse (90% zero)": lambda: torch.randn(1024, 128, device=device)
        * (torch.rand(1024, 128, device=device) > 0.9).float(),
        "Outlier (1% at 10x)": lambda: _make_outlier_tensor(device),
    }

    print(
        f"\n{'Distribution':<25} {'Method':<15} {'MSE':>12} {'SNR (dB)':>10} {'MaxErr':>10}"
    )
    print("-" * 72)

    for dist_name, gen_fn in distributions.items():
        tensor = gen_fn()

        # V1 AXS-6 (uniform grid)
        axs_v1 = quantize(tensor, block_size=32, rounding="nearest")
        recon_v1 = dequantize(axs_v1)
        err_v1 = (tensor - recon_v1).float()
        mse_v1 = err_v1.pow(2).mean().item()
        snr_v1 = 10 * math.log10(
            tensor.float().pow(2).mean().item() / max(mse_v1, 1e-45)
        )
        max_v1 = err_v1.abs().max().item()

        # V2 AXS-6 (NF5 + percentile clip)
        axs_v2 = quantize_v2(
            tensor, block_size=32, rounding="nearest", use_nf5=True, clip_percentile=99.9
        )
        recon_v2 = dequantize_v2(axs_v2, use_nf5=True)
        err_v2 = (tensor - recon_v2).float()
        mse_v2 = err_v2.pow(2).mean().item()
        snr_v2 = 10 * math.log10(
            tensor.float().pow(2).mean().item() / max(mse_v2, 1e-45)
        )
        max_v2 = err_v2.abs().max().item()

        # FP8 E4M3
        recon_fp8 = fp8_e4m3_fake_quantize(tensor)
        err_fp8 = (tensor - recon_fp8).float()
        mse_fp8 = err_fp8.pow(2).mean().item()
        snr_fp8 = 10 * math.log10(
            tensor.float().pow(2).mean().item() / max(mse_fp8, 1e-45)
        )
        max_fp8 = err_fp8.abs().max().item()

        print(f"{dist_name:<25} {'AXS-6 v1':<15} {mse_v1:>12.8f} {snr_v1:>10.2f} {max_v1:>10.6f}")
        print(f"{'':<25} {'AXS-6 v2':<15} {mse_v2:>12.8f} {snr_v2:>10.2f} {max_v2:>10.6f}")
        print(f"{'':<25} {'FP8 E4M3':<15} {mse_fp8:>12.8f} {snr_fp8:>10.2f} {max_fp8:>10.6f}")

        # Highlight winner
        best_snr = max(snr_v1, snr_v2, snr_fp8)
        winner = "v1" if best_snr == snr_v1 else ("v2" if best_snr == snr_v2 else "fp8")
        v2_improvement = snr_v2 - snr_v1
        print(
            f"{'':<25} {'→ Winner:':<15} {winner:>12}  (v2 {v2_improvement:+.2f} dB vs v1)"
        )
        print()


def _make_outlier_tensor(device: torch.device) -> torch.Tensor:
    """Create tensor with 1% outliers at 10× normal magnitude."""
    t = torch.randn(1024, 128, device=device)
    mask = torch.rand_like(t) < 0.01
    t[mask] *= 10.0
    return t


# ---------------------------------------------------------------------------
# Test 2: Training Benchmark
# ---------------------------------------------------------------------------


def test_training(
    num_steps: int = 200,
    batch_size: int = 32,
) -> None:
    """Train MiniGPT with FP32, v1, v2, and measure convergence + speed."""
    print("=" * 70)
    print("TEST 2: TRAINING BENCHMARK (MiniGPT)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BenchConfig()

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps: {num_steps}, Batch: {batch_size}, Seq: {cfg.context_len}")
    print()

    # Pre-generate data for fair comparison
    torch.manual_seed(42)
    data_x = torch.randint(0, cfg.vocab_size, (num_steps + 5, batch_size, cfg.context_len))
    data_y = torch.randint(0, cfg.vocab_size, (num_steps + 5, batch_size, cfg.context_len))

    results = {}

    for mode_name in ["FP32", "AXS-6 v1", "AXS-6 v2"]:
        print(f"--- {mode_name} ---")

        # Create model with same init
        torch.manual_seed(123)
        base_model = MiniGPT(cfg)

        if mode_name == "FP32":
            model = base_model
        elif mode_name == "AXS-6 v1":
            model = convert_to_axs(base_model, block_size=32)
        else:  # v2
            model = convert_to_axs_v2(base_model, block_size=32)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  Params: {nparams:,}")

        # Warmup
        model.train()
        for i in range(5):
            x = data_x[i].to(device)
            y = data_y[i].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()
        losses = []
        t0 = time.perf_counter()

        for step in range(num_steps):
            x = data_x[step + 5].to(device)
            y = data_y[step + 5].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms_per_step = elapsed / num_steps * 1000

        final_loss = sum(losses[-20:]) / 20
        final_ppl = math.exp(min(final_loss, 20))

        # Early convergence: first step where loss < initial_loss * 0.8
        initial_loss = losses[0]
        convergence_step = num_steps
        threshold = initial_loss * 0.8
        for i, l in enumerate(losses):
            if l < threshold:
                convergence_step = i
                break

        results[mode_name] = {
            "ms_per_step": ms_per_step,
            "final_loss": final_loss,
            "final_ppl": final_ppl,
            "convergence_step": convergence_step,
            "losses": losses,
        }

        print(f"  Time: {ms_per_step:.2f} ms/step")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final PPL: {final_ppl:.2f}")
        print(f"  80% convergence at step: {convergence_step}")
        print()

        # Free memory
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary comparison
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<15} {'ms/step':>10} {'Loss':>10} {'PPL':>10} {'Conv@80%':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(
            f"{name:<15} {r['ms_per_step']:>10.2f} {r['final_loss']:>10.4f} "
            f"{r['final_ppl']:>10.2f} {r['convergence_step']:>10}"
        )

    # V2 vs V1 delta
    if "AXS-6 v1" in results and "AXS-6 v2" in results:
        v1 = results["AXS-6 v1"]
        v2 = results["AXS-6 v2"]
        fp32 = results.get("FP32", v1)
        print(f"\n--- V2 vs V1 Improvement ---")
        ppl_delta = v1["final_ppl"] - v2["final_ppl"]
        print(f"  PPL reduction: {ppl_delta:+.2f} ({ppl_delta / v1['final_ppl'] * 100:+.1f}%)")
        loss_delta = v1["final_loss"] - v2["final_loss"]
        print(f"  Loss reduction: {loss_delta:+.6f}")
        speed_delta = v2["ms_per_step"] - v1["ms_per_step"]
        print(f"  Speed overhead: {speed_delta:+.2f} ms/step ({speed_delta / v1['ms_per_step'] * 100:+.1f}%)")

        # Gap to FP32
        v1_gap = abs(v1["final_ppl"] - fp32["final_ppl"])
        v2_gap = abs(v2["final_ppl"] - fp32["final_ppl"])
        print(f"\n--- Gap to FP32 ---")
        print(f"  V1 PPL gap: {v1_gap:.2f}")
        print(f"  V2 PPL gap: {v2_gap:.2f}")
        print(f"  V2 closes {max(0, v1_gap - v2_gap):.2f} PPL of the gap ({max(0, (v1_gap - v2_gap) / max(v1_gap, 0.01) * 100):.1f}%)")


# ---------------------------------------------------------------------------
# Test 3: Per-Layer Error Analysis
# ---------------------------------------------------------------------------


def test_per_layer_error() -> None:
    """Compare v1 vs v2 quantization error for each layer's weights."""
    print("\n" + "=" * 70)
    print("TEST 3: PER-LAYER WEIGHT QUANTIZATION ERROR")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BenchConfig()
    model = MiniGPT(cfg).to(device)

    print(
        f"\n{'Layer':<40} {'v1 SNR (dB)':>12} {'v2 SNR (dB)':>12} {'Delta':>8} {'Winner':>8}"
    )
    print("-" * 80)

    total_v1_mse = 0.0
    total_v2_mse = 0.0
    total_numel = 0

    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue  # skip biases

        w = param.data

        # V1 error
        axs_v1 = quantize(w, block_size=32, rounding="nearest")
        recon_v1 = dequantize(axs_v1)
        err_v1 = (w - recon_v1).float()
        mse_v1 = err_v1.pow(2).mean().item()
        signal_power = w.float().pow(2).mean().item()
        snr_v1 = 10 * math.log10(signal_power / max(mse_v1, 1e-45))

        # V2 error
        axs_v2 = quantize_v2(w, block_size=32, rounding="nearest", use_nf5=True, clip_percentile=99.9)
        recon_v2 = dequantize_v2(axs_v2, use_nf5=True)
        err_v2 = (w - recon_v2).float()
        mse_v2 = err_v2.pow(2).mean().item()
        snr_v2 = 10 * math.log10(signal_power / max(mse_v2, 1e-45))

        delta = snr_v2 - snr_v1
        winner = "v2" if delta > 0 else "v1"
        print(f"{name:<40} {snr_v1:>12.2f} {snr_v2:>12.2f} {delta:>+8.2f} {winner:>8}")

        total_v1_mse += mse_v1 * w.numel()
        total_v2_mse += mse_v2 * w.numel()
        total_numel += w.numel()

    # Aggregate
    avg_v1_mse = total_v1_mse / max(total_numel, 1)
    avg_v2_mse = total_v2_mse / max(total_numel, 1)
    if avg_v1_mse > 0:
        improvement = (1 - avg_v2_mse / avg_v1_mse) * 100
    else:
        improvement = 0.0
    print(f"\nAggregate weighted MSE: v1={avg_v1_mse:.10f}, v2={avg_v2_mse:.10f}")
    print(f"V2 MSE reduction: {improvement:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 70)
    print("  AXS-6 V1 vs V2 — COMPREHENSIVE A/B BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    test_quantization_error()
    test_training(num_steps=200, batch_size=32)
    test_per_layer_error()

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
