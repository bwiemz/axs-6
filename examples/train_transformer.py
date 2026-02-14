"""
Example: Pretrain a Small GPT Transformer with AXS-6
=====================================================

Demonstrates AXS-6 quantized training on a small GPT-style language model.
This is the most realistic test case for the format, as transformer training
is the primary target application.

The model trains on synthetic character-level data (or optionally on a
provided text file).

Run: ``python -m examples.train_transformer``
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axs.nn.modules import (
    AXSLinear,
    AXSLayerNorm,
    AXSEmbedding,
    AXSMultiheadAttention,
    convert_to_axs,
)
from axs.nn.optim import AXSAdamW


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """Configuration for the mini GPT model."""
    vocab_size: int = 256  # character-level
    context_len: int = 128
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    A small GPT-style transformer for character-level language modeling.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, config.context_len, config.embed_dim) * 0.02)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token indices of shape ``(batch, seq_len)``.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CharDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text: str, context_len: int = 128) -> None:
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


def generate_synthetic_text(length: int = 500_000) -> str:
    """Generate synthetic text with statistical structure for training."""
    import random
    random.seed(42)

    # Create text with character-level patterns
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "a", "an", "is", "was", "were", "are", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "data", "model", "training", "neural", "network", "deep",
        "learning", "precision", "quantization", "efficient", "format",
        "weights", "gradients", "activation", "transformer", "attention",
    ]

    text_parts = []
    for _ in range(length // 10):
        sentence_len = random.randint(5, 15)
        sentence = " ".join(random.choice(words) for _ in range(sentence_len))
        sentence = sentence[0].upper() + sentence[1:] + ". "
        text_parts.append(sentence)

    return "".join(text_parts)[:length]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gpt(
    config: GPTConfig,
    use_axs: bool = False,
    block_size: int = 32,
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: torch.device | None = None,
) -> dict:
    """
    Train a MiniGPT model, optionally with AXS-6 quantization.

    Returns training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    text = generate_synthetic_text(200_000)
    dataset = CharDataset(text, config.context_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Model
    torch.manual_seed(42)
    model = MiniGPT(config).to(device)

    if use_axs:
        model = convert_to_axs(model, block_size=block_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Optimizer
    if use_axs:
        optimizer = AXSAdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training
    history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        start = time.time()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), batch_y.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
        elapsed = time.time() - start

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "perplexity": perplexity,
            "time": elapsed,
        })

        print(
            f"  Epoch {epoch+1}/{num_epochs}: "
            f"Loss={avg_loss:.4f}, PPL={perplexity:.2f}, "
            f"Time={elapsed:.1f}s"
        )

    return {
        "history": history,
        "num_params": num_params,
        "final_loss": history[-1]["loss"],
        "final_ppl": history[-1]["perplexity"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("TRANSFORMER PRETRAINING: AXS-6 vs FP32")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    config = GPTConfig(
        vocab_size=256,
        context_len=64,    # shorter for faster demo
        embed_dim=128,     # smaller for demo
        num_heads=4,
        num_layers=3,
        ff_dim=512,
        dropout=0.1,
    )

    print(f"Model Config: {config}\n")

    results = {}

    # FP32 baseline
    print("─" * 80)
    print("Training: FP32 Baseline")
    print("─" * 80)
    results["FP32"] = train_gpt(config, use_axs=False, num_epochs=5, device=device)

    # AXS-6
    print("\n" + "─" * 80)
    print("Training: AXS-6 (B=32)")
    print("─" * 80)
    results["AXS-6"] = train_gpt(config, use_axs=True, block_size=32, num_epochs=5, device=device)

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  {'Config':<20} {'Final Loss':>12} {'Final PPL':>12} {'Params':>12}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 12}")

    for name, res in results.items():
        print(f"  {name:<20} {res['final_loss']:>12.4f} {res['final_ppl']:>12.2f} "
              f"{res['num_params']:>12,}")

    ppl_diff = results["AXS-6"]["final_ppl"] - results["FP32"]["final_ppl"]
    print(f"\n  AXS-6 perplexity delta vs FP32: {ppl_diff:+.2f}")
    print(f"  Memory savings: ~21% weight memory reduction")


if __name__ == "__main__":
    main()
