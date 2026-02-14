"""
AXS-6 Backend Benchmark: Triton vs Compiled vs Eager
=====================================================

Compares the three AXS-6 backend implementations across different tensor
sizes and in an end-to-end training scenario.

Run::

    python benchmarks/benchmark_triton.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", r"C:\tmp\ti")

import torch
import torch.nn as nn
import torch.nn.functional as F

from axs.unified.backend import (
    accelerated_fake_quantize,
    set_backend,
)
from axs.unified.modules_unified import convert_to_axs_unified
from axs.unified.quantize_unified import fused_fake_quantize
from axs.unified.triton_kernels import has_triton, triton_fused_fake_quantize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bench_fq(
    fn: callable,
    x: torch.Tensor,
    warmup: int = 20,
    iters: int = 100,
) -> float:
    """Benchmark a fake-quantize function.  Returns ms per call."""
    for _ in range(warmup):
        fn(x, 32, "nearest")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, 32, "nearest")
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


class MiniGPT(nn.Module):
    """Tiny GPT-style model for training benchmarks."""

    def __init__(
        self, vocab: int = 1000, dim: int = 256, heads: int = 4, layers: int = 4
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(128, dim)
        enc_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.0, batch_first=True
        )
        self.tf = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.head(self.tf(h, mask=mask, is_causal=True))


def train_run(backend_name: str, steps: int = 50) -> tuple[float, float]:
    """Run a short training loop and return (ms_per_step, final_loss)."""
    model = MiniGPT().cuda()
    model = convert_to_axs_unified(model)
    set_backend(backend_name)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randint(0, 1000, (8, 64), device="cuda")

    # Warmup
    for _ in range(10):
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, 1000), data.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, 1000), data.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    ms_per_step = (time.perf_counter() - t0) / steps * 1000
    return ms_per_step, loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    print("=" * 64)
    print("AXS-6 Backend Benchmark: Triton vs Compiled vs Eager")
    print("=" * 64)
    print(f"GPU:     {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {'available' if has_triton() else 'not available'}")
    print()

    # --- Fake-quantize microbenchmark ---
    print("--- Fake-Quantize Latency (ms) ---")
    print(f"{'Shape':>14}  {'Triton':>8}  {'Compiled':>8}  {'Eager':>8}  "
          f"{'vs Eager':>9}  {'vs Compiled':>12}")
    print("-" * 70)

    shapes = [
        ("256x256", (256, 256)),
        ("1024x1024", (1024, 1024)),
        ("4096x4096", (4096, 4096)),
        ("8192x4096", (8192, 4096)),
    ]

    for label, shape in shapes:
        x = torch.randn(*shape, device="cuda")

        eager_ms = bench_fq(fused_fake_quantize, x)

        set_backend("compiled")
        compiled_ms = bench_fq(accelerated_fake_quantize, x)

        triton_ms = bench_fq(triton_fused_fake_quantize, x)

        print(
            f"{label:>14}  {triton_ms:>8.3f}  {compiled_ms:>8.3f}  "
            f"{eager_ms:>8.3f}  {eager_ms / triton_ms:>8.1f}x  "
            f"{compiled_ms / triton_ms:>11.1f}x"
        )

    # --- Training benchmark ---
    print()
    print("--- End-to-end Training (MiniGPT, 50 steps) ---")
    print(f"{'Backend':>10}  {'ms/step':>8}  {'Loss':>8}  {'vs Eager':>9}")
    print("-" * 42)

    results = {}
    for name in ["triton", "compiled", "eager"]:
        ms, loss = train_run(name)
        results[name] = ms
        speedup = results.get("eager", ms) / ms
        print(f"{name:>10}  {ms:>8.2f}  {loss:>8.4f}  {speedup:>8.1f}x")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
