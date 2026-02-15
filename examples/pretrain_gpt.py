"""
Pretrain a GPT Transformer with AXS-6 Mixed-Precision
=======================================================

Demonstrates a realistic pretraining setup with:

  - AXS-6 NF5 + BF16 mixed-precision (tensor cores)
  - Cosine LR schedule with linear warmup
  - Checkpoint save/resume for crash recovery
  - Weight tying (embedding ↔ LM head)
  - Gradient clipping + adaptive loss scaling
  - Per-step logging with throughput metrics

The model trains on synthetic character-level data (or a text file).

Run::

    python -m examples.pretrain_gpt

Resume from checkpoint::

    python -m examples.pretrain_gpt --resume checkpoints/step_500.pt
"""

from __future__ import annotations

import argparse
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

from axs.unified.mixed_precision import (
    AXSLinearMixedPrecision,
    convert_to_axs_mixed_precision,
)
from axs.unified.modules_unified import (
    AXSEmbeddingUnified,
    AXSLayerNormUnified,
)
from axs.unified.training_unified import AXSTrainingPipelineUnified


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 256       # character-level
    context_len: int = 128
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    # AXS-6
    block_size: int = 32


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    num_epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100     # LR warmup (linear ramp)
    quant_warmup_steps: int = 50  # quantisation warmup (FP32 first N steps)
    max_grad_norm: float = 1.0
    checkpoint_every: int = 500   # save checkpoint every N steps
    log_every: int = 50           # log every N steps
    data_length: int = 500_000
    checkpoint_dir: str = "checkpoints"


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
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            attn_mask=causal_mask,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """GPT-style transformer for character-level language modelling."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.context_len, config.embed_dim) * 0.02,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)],
        )
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embedding
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
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CharDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text: str, context_len: int = 128) -> None:
        self.data = torch.tensor(
            [ord(c) % 256 for c in text], dtype=torch.long,
        )
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


def generate_synthetic_text(length: int = 500_000) -> str:
    """Generate synthetic text with statistical structure."""
    import random
    random.seed(42)

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

    text_parts: list[str] = []
    for _ in range(length // 10):
        sentence_len = random.randint(5, 15)
        sentence = " ".join(random.choice(words) for _ in range(sentence_len))
        sentence = sentence[0].upper() + sentence[1:] + ". "
        text_parts.append(sentence)

    return "".join(text_parts)[:length]


# ---------------------------------------------------------------------------
# LR Schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# AXS-6 Model Conversion (with weight-tying awareness)
# ---------------------------------------------------------------------------


def convert_gpt_to_axs(
    model: MiniGPT,
    block_size: int = 32,
) -> MiniGPT:
    """
    Convert MiniGPT to AXS-6 BF16 mixed-precision.

    Handles weight tying: the LM head and embedding share a weight tensor.
    We convert all nn.Linear layers to AXSLinearMixedPrecision, but skip
    the tied ``head`` layer — it reuses ``token_embed.weight`` directly.
    After conversion we re-tie.
    """
    # Step 1: Remember if weight tying is active
    is_tied = model.head.weight.data_ptr() == model.token_embed.weight.data_ptr()

    # Step 2: Convert Linear → AXSLinearMixedPrecision (skip head)
    converted = convert_to_axs_mixed_precision(
        model,
        block_size=block_size,
        skip_layers={"head"} if is_tied else set(),
        inplace=True,
    )

    # Step 3: Re-tie if needed — the LM head is a plain nn.Linear that
    # shares its weight Parameter with the embedding.  After conversion
    # the embedding is still nn.Embedding (not converted by mixed-precision
    # converter), so we just re-tie.
    if is_tied:
        converted.head.weight = converted.token_embed.weight

    return converted


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(
    gpt_config: GPTConfig,
    train_config: TrainConfig,
    resume_path: str | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Full pretraining run with AXS-6 BF16 mixed-precision.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────
    text = generate_synthetic_text(train_config.data_length)
    dataset = CharDataset(text, gpt_config.context_len)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    total_steps = train_config.num_epochs * len(loader)
    print(f"  Dataset: {len(dataset):,} samples, {len(loader)} batches/epoch")
    print(f"  Total steps: {total_steps:,}")

    # ── Model ─────────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = MiniGPT(gpt_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} (FP32 baseline)")

    # Convert to AXS-6 BF16 mixed-precision
    model = convert_gpt_to_axs(model, block_size=gpt_config.block_size)
    print(f"  Converted to AXS-6 BF16 mixed-precision (block_size={gpt_config.block_size})")

    # Verify weight tying
    assert model.head.weight.data_ptr() == model.token_embed.weight.data_ptr(), (
        "Weight tying broken after conversion!"
    )
    print(f"  Weight tying: OK (head ↔ token_embed)")

    # ── Optimiser + Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=train_config.min_lr / train_config.lr,
    )

    # ── Pipeline ──────────────────────────────────────────────────────
    pipeline = AXSTrainingPipelineUnified(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=train_config.quant_warmup_steps,
        max_grad_norm=train_config.max_grad_norm,
        loss_scale_init=2.0 ** 16,
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    if resume_path is not None:
        extra = pipeline.load_checkpoint(resume_path, map_location=device)
        start_epoch = extra.get("epoch", 0)
        print(f"  Resumed from {resume_path} (step {pipeline._total_steps}, epoch {start_epoch})")

    # ── Train ─────────────────────────────────────────────────────────
    ckpt_dir = Path(train_config.checkpoint_dir)
    history: list[dict] = []
    tokens_per_batch = train_config.batch_size * gpt_config.context_len
    global_step = pipeline._total_steps

    print()
    print(f"{'Step':>7} {'Epoch':>5} {'Loss':>8} {'PPL':>8} "
          f"{'LR':>10} {'Grad':>8} {'Tok/s':>10} {'Warmup':>6}")
    print("─" * 75)

    for epoch in range(start_epoch, train_config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        t_epoch = time.time()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            t0 = time.time()

            def loss_fn(logits: torch.Tensor, _inputs: torch.Tensor) -> torch.Tensor:
                return F.cross_entropy(
                    logits.float().view(-1, gpt_config.vocab_size),
                    batch_y.view(-1),
                )

            stats = pipeline.training_step(batch_x, loss_fn)
            dt = time.time() - t0

            global_step = stats["step"]
            epoch_loss += stats["loss"]
            epoch_batches += 1

            # Logging
            if global_step % train_config.log_every == 0:
                ppl = math.exp(min(stats["loss"], 20))
                tok_s = tokens_per_batch / dt
                mark = "Y" if stats["warmup"] else ""
                print(
                    f"{global_step:>7d} {epoch+1:>5d} {stats['loss']:>8.4f} "
                    f"{ppl:>8.2f} {stats['lr']:>10.2e} "
                    f"{stats['grad_norm']:>8.4f} {tok_s:>10,.0f} {mark:>6s}"
                )

            # Checkpoint
            if global_step % train_config.checkpoint_every == 0:
                pipeline.save_checkpoint(
                    ckpt_dir / f"step_{global_step}.pt",
                    extra={"epoch": epoch},
                )

        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_ppl = math.exp(min(avg_loss, 20))
        epoch_time = time.time() - t_epoch
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "perplexity": avg_ppl,
            "time": epoch_time,
        })
        print(f"  ── Epoch {epoch+1} done: loss={avg_loss:.4f}, "
              f"ppl={avg_ppl:.2f}, time={epoch_time:.1f}s ──")

    # Final checkpoint
    pipeline.save_checkpoint(
        ckpt_dir / "final.pt",
        extra={"epoch": train_config.num_epochs},
    )

    print()
    print("Training complete.")
    print(f"  Final loss:  {history[-1]['loss']:.4f}")
    print(f"  Final PPL:   {history[-1]['perplexity']:.2f}")
    print(f"  Checkpoints: {ckpt_dir.resolve()}")

    return {
        "history": history,
        "num_params": num_params,
        "total_steps": global_step,
        "stats": pipeline.stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AXS-6 GPT Pretraining")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--data-length", type=int, default=500_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    print("=" * 75)
    print("AXS-6 GPT PRETRAINING (BF16 Mixed-Precision)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'})")

    gpt_config = GPTConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        context_len=args.context_len,
    )
    train_config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_length=args.data_length,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"  Model:  {gpt_config}")
    print(f"  Train:  {train_config}")
    print()

    train(gpt_config, train_config, resume_path=args.resume, device=device)


if __name__ == "__main__":
    main()
