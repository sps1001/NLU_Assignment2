"""
Task-2: Hyperparameter Heatmaps
CSL 7640 - NLU Assignment 2, Problem 1
Author: B23CS1061

Generates 6 coloured heatmaps (3 CBOW + 3 Skip-gram), one per negative-sample
value, showing Avg NN Similarity across Embedding Dim × Window Size.
Also saves a combined 2×3 figure for the report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = "outputs"
CSV_PATH   = os.path.join(OUTPUT_DIR, "experiment_results.csv")

df = pd.read_csv(CSV_PATH)

NEG_VALUES  = [5, 10, 15]
DIMS        = [50, 100, 200]
WINDOWS     = [2, 5, 10]
ARCHS       = ["CBOW", "SkipGram"]

# Colour maps: warm for CBOW, cool for Skip-gram
CMAPS = {"CBOW": "YlOrRd", "SkipGram": "YlGnBu"}
ARCH_LABELS = {"CBOW": "CBOW", "SkipGram": "Skip-gram"}


def build_heatmap_matrix(arch: str, neg: int) -> np.ndarray:
    """Extract 3×3 similarity matrix (rows=dim, cols=window) for given arch & neg."""
    subset = df[(df["Architecture"] == arch) & (df["Negative Samples"] == neg)]
    mat = np.zeros((3, 3))
    for i, dim in enumerate(DIMS):
        for j, win in enumerate(WINDOWS):
            val = subset[(subset["Embedding Dim"] == dim) &
                         (subset["Window Size"] == win)]["Avg NN Sim"].values
            mat[i, j] = val[0] if len(val) else np.nan
    return mat


def plot_single(ax, mat: np.ndarray, title: str, cmap: str,
                vmin: float, vmax: float, show_cbar: bool = True):
    """Draw one annotated heatmap on the given axes."""
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Annotate each cell
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            # White text on dark cells, black on light
            brightness = (val - vmin) / (vmax - vmin) if (vmax > vmin) else 0.5
            color = "white" if brightness > 0.6 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"win={w}" for w in WINDOWS], fontsize=9)
    ax.set_yticklabels([f"dim={d}" for d in DIMS], fontsize=9)
    ax.set_xlabel("Context Window Size", fontsize=9)
    ax.set_ylabel("Embedding Dimension", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Avg NN Similarity", fontsize=8)

    return im


# ── 1. Individual heatmaps (6 files) ──────────────────────────────────────────
for arch in ARCHS:
    # Compute global vmin/vmax across all neg for this arch (consistent colour scale)
    all_vals = df[df["Architecture"] == arch]["Avg NN Sim"]
    vmin, vmax = all_vals.min() - 0.001, all_vals.max() + 0.001

    for neg in NEG_VALUES:
        mat = build_heatmap_matrix(arch, neg)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        plot_single(ax, mat,
                    title=f"{ARCH_LABELS[arch]}  —  Negative Samples = {neg}",
                    cmap=CMAPS[arch], vmin=vmin, vmax=vmax)
        plt.tight_layout()
        fname = f"task2_heatmap_{arch.lower()}_neg{neg}.png"
        path  = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


# ── 2. Combined 2×3 figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Word2Vec Hyperparameter Search — Avg NN Similarity\n"
    "Embedding Dim × Window Size  |  rows: CBOW / Skip-gram  |  cols: neg = 5 / 10 / 15",
    fontsize=12, fontweight="bold", y=1.01
)

for row, arch in enumerate(ARCHS):
    all_vals = df[df["Architecture"] == arch]["Avg NN Sim"]
    vmin, vmax = all_vals.min() - 0.001, all_vals.max() + 0.001

    for col, neg in enumerate(NEG_VALUES):
        mat = build_heatmap_matrix(arch, neg)
        show_cbar = (col == 2)   # only rightmost column gets colour bar
        plot_single(axes[row][col], mat,
                    title=f"{ARCH_LABELS[arch]}  |  neg={neg}",
                    cmap=CMAPS[arch], vmin=vmin, vmax=vmax,
                    show_cbar=show_cbar)

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, "task2_heatmaps_combined.png")
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Combined figure → {combined_path}")
print("Done.")
