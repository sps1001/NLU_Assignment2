"""
Task-1 & Task-2: Training all three models
CSL 7640 - NLU Assignment 2, Problem 2
Author: B23CS1061

Trains VanillaRNN, BidirectionalLSTM, and AttentionRNN on the Indian names
dataset. For each model:
  - Reports architecture and trainable parameter count
  - Trains with cross-entropy loss (teacher-forced)
  - Saves the best checkpoint (lowest validation loss)
  - Saves per-epoch loss curves to outputs/

Usage:
    python train.py
"""

import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_names, build_vocab, get_dataloader, EOS_TOKEN, PAD_TOKEN
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = "data/TrainingNames.txt"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters (shared across all models for fair comparison)
EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT   = 0.1    # fraction of data used for validation
SEED        = 42

# Model hyperparameters
EMBED_DIM        = 64
HIDDEN_SIZE      = 256   # VanillaRNN and AttentionRNN
BLSTM_HIDDEN     = 128   # BLSTM uses 128 so two LSTMs stay param-comparable to others
NUM_LAYERS       = 2
DROPOUT          = 0.3

# Generation params for quick sanity check after training
GEN_TEMP    = 0.8
GEN_COUNT   = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device() -> torch.device:
    """Prefer MPS (Apple Silicon), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_names(names, val_split, seed):
    """Reproducible train/val split."""
    import random
    rng = random.Random(seed)
    names = names[:]
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_split))
    return names[n_val:], names[:n_val]


def compute_loss_batch(
    model,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    criterion: nn.CrossEntropyLoss,
    is_blstm: bool,
    is_attn: bool,
    val_mode: bool = False,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for one batch.

    VanillaRNN / AttentionRNN:
        input  = seq[:-1]  (SOS … last_char)
        target = seq[1:]   (first_char … EOS)
        loss   = forward CE only

    BLSTM (ELMo BiLM):
        Training  : loss = forward_CE + backward_CE  (joint BiLM objective)
        Validation: loss = forward_CE only
          → This makes val_loss directly comparable to the other two models,
            since all three report single-direction CE on the same target.
          → Checkpointing also uses forward_CE so the saved model maximises
            left-to-right generation quality, not the combined BiLM objective.
    """
    seqs, _ = batch
    seqs = seqs.to(device)          # (B, T)

    if is_blstm:
        # BiLM: model returns (fwd_logits, bwd_logits) both (B, T-1, V)
        fwd_logits, bwd_logits = model(seqs)

        # Forward target: x[1..T]
        fwd_tgt = seqs[:, 1:]                   # (B, T-1)
        # Backward target: reversed x[0..T-2]
        bwd_tgt = seqs[:, :-1].flip(dims=[1])   # (B, T-1)

        B, T, V = fwd_logits.shape
        fwd_loss = criterion(fwd_logits.reshape(B*T, V), fwd_tgt.reshape(B*T))

        if val_mode:
            # Validation: forward CE only — apples-to-apples with other models
            return fwd_loss

        # Training: joint BiLM loss drives the shared embedding to encode
        # both forward and backward context, enriching representations
        bwd_loss = criterion(bwd_logits.reshape(B*T, V), bwd_tgt.reshape(B*T))
        return fwd_loss + bwd_loss

    elif is_attn:
        # AttentionRNN: takes full sequence, returns (B, T-1, V)
        logits = model(seqs)                    # (B, T-1, V)
        tgt    = seqs[:, 1:]                    # (B, T-1)
        B, T, V = logits.shape
        return criterion(logits.reshape(B*T, V), tgt.reshape(B*T))

    else:
        # VanillaRNN: takes inp=(B,T-1), returns (B,T-1,V)
        inp    = seqs[:, :-1]                   # (B, T-1)
        tgt    = seqs[:, 1:]
        logits, _ = model(inp)
        B, T, V = logits.shape
        return criterion(logits.reshape(B*T, V), tgt.reshape(B*T))


def train_model(model, name: str, train_names, val_names, char2idx, device):
    """
    Full training loop for one model.
    Returns a dict of training metadata (losses, params, time).
    """
    print(f"\n{'='*60}")
    print(f" Training: {name}")
    print(f" Parameters: {count_parameters(model):,}")
    print(f" Device: {device}")
    print(f"{'='*60}")

    model.to(device)

    train_loader = get_dataloader(train_names, char2idx, BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(val_names,   char2idx, BATCH_SIZE, shuffle=False)

    # Ignore PAD tokens in loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Reduce LR on plateau to help convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    is_blstm = isinstance(model, BidirectionalLSTM)
    is_attn  = isinstance(model, AttentionRNN)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    ckpt_path = os.path.join(MODELS_DIR, f"{name}.pt")

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_batch(model, batch, device, criterion, is_blstm, is_attn)
            loss.backward()
            # Gradient clipping prevents exploding gradients in RNNs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # ── Validate ──
        # val_mode=True ensures BLSTM reports only forward CE (comparable to others)
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss_batch(model, batch, device, criterion,
                                          is_blstm, is_attn, val_mode=True)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)

        scheduler.step(avg_val)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"best_val={best_val_loss:.4f}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s  |  best_val_loss={best_val_loss:.4f}")
    print(f"Checkpoint → {ckpt_path}")

    # Save loss curves plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses,   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{name} — Training Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_loss_curve.png"), dpi=150)
    plt.close(fig)

    return {
        "name"          : name,
        "params"        : count_parameters(model),
        "best_val_loss" : round(best_val_loss, 5),
        "train_time_s"  : round(elapsed, 1),
        "train_losses"  : train_losses,
        "val_losses"    : val_losses,
        "checkpoint"    : ckpt_path,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"Device: {device}")

    # Load dataset
    names = load_names(DATA_FILE)
    print(f"Loaded {len(names)} unique names.")

    char2idx, idx2char = build_vocab(names)
    vocab_size = len(char2idx)
    print(f"Vocabulary size: {vocab_size} characters")

    # Save vocab for use in evaluation / generation scripts
    with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f)

    train_names, val_names = split_names(names, VAL_SPLIT, SEED)
    print(f"Train: {len(train_names)}  Val: {len(val_names)}")

    # Instantiate all three models
    # BLSTM uses hidden_size=128 (two full LSTMs → ~472K params, comparable to others)
    # VanillaRNN and AttentionRNN use hidden_size=256 (~223K and ~362K params)
    model_configs = [
        ("VanillaRNN",   VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE,  NUM_LAYERS, DROPOUT)),
        ("BLSTM",        BidirectionalLSTM(vocab_size, EMBED_DIM, BLSTM_HIDDEN, NUM_LAYERS, DROPOUT)),
        ("AttentionRNN", AttentionRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE,  NUM_LAYERS, DROPOUT)),
    ]

    # Print architecture summaries
    print("\n── Model Parameter Counts ──────────────────────────────────")
    for mname, model in model_configs:
        print(f"  {mname:20s}: {count_parameters(model):>10,} parameters")
    print("──────────────────────────────────────────────────────────\n")

    all_meta = {}
    for mname, model in model_configs:
        meta = train_model(model, mname, train_names, val_names, char2idx, device)
        all_meta[mname] = meta

        # Quick sanity generation after training
        model.load_state_dict(torch.load(meta["checkpoint"], map_location=device, weights_only=True))
        print(f"\n  Sample names from {mname}:")
        for _ in range(GEN_COUNT):
            tokens = model.generate(device, temperature=GEN_TEMP)
            name_str = "".join(idx2char.get(t, "?") for t in tokens)
            print(f"    {name_str.capitalize()}")

    # Save training summary (losses stripped to keep file small)
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("train_losses", "val_losses")}
        for k, v in all_meta.items()
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary → {OUTPUT_DIR}/training_summary.json")

    # Combined loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training & Validation Loss — All Models", fontsize=12, fontweight="bold")
    colors = [("#E63946", "#457B9D"), ("#2A9D8F", "#E9C46A"), ("#9B5DE5", "#F4A261")]
    for ax, (mname, meta), (tc, vc) in zip(axes, all_meta.items(), colors):
        ax.plot(meta["train_losses"], color=tc, label="Train")
        ax.plot(meta["val_losses"],   color=vc, label="Val")
        ax.set_title(mname)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "all_loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Combined loss curves → {OUTPUT_DIR}/all_loss_curves.png")

    print("\nTraining complete. Next → run evaluate.py for Task-2 metrics.")


if __name__ == "__main__":
    main()
