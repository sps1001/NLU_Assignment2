"""
Task-4: Visualization of Word Embeddings
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Projects word embeddings from the best CBOW and Skip-gram models into 2D space
using both PCA and t-SNE. Words are colour-coded by semantic category so that
clustering behaviour is visually interpretable.

Word groups are chosen to span the main topical areas of the IIT-Jodhpur corpus:
  1. Academic Roles    — people and roles in academic institutions
  2. Degree Levels     — degree programme labels
  3. Academic Activities — learning and research processes
  4. STEM & Tech       — core subject areas in the corpus
  5. AI / ML Domain    — specialised computing vocabulary (heavy in CSE/AI&DS data)
  6. Course Structure  — semester/curriculum administrative words
  7. Places & Institute — geographic and institutional references

Outputs (all saved to outputs/):
  task4_pca_cbow.png        – PCA projection, CBOW model
  task4_pca_skipgram.png    – PCA projection, Skip-gram model
  task4_tsne_cbow.png       – t-SNE projection, CBOW model
  task4_tsne_skipgram.png   – t-SNE projection, Skip-gram model
  task4_combined.png        – 2×2 grid comparing both methods × both models
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"
BEST_MODELS_FILE = os.path.join(OUTPUT_DIR, "best_models.json")

# Semantic word groups for visualisation — all verified in-vocab on the expanded corpus
WORD_GROUPS = {
    "Academic Roles"      : ["professor", "faculty", "researcher", "student", "teaching"],
    "Degree Levels"       : ["undergraduate", "postgraduate", "phd", "mtech", "btech"],
    "Academic Activities" : ["research", "thesis", "admission", "scholarship", "exam"],
    "STEM & Tech"         : ["engineering", "science", "mathematics", "technology", "computing"],
    "AI / ML Domain"      : ["machine", "learning", "neural", "algorithm", "data"],
    "Course Structure"    : ["semester", "course", "curriculum", "credit", "department"],
    "Places & Institute"  : ["jodhpur", "delhi", "jaipur", "rajasthan", "india"],
}

# Colourblind-friendly palette — one colour per group (7 groups)
PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#9B5DE5", "#F4A261", "#06D6A0"]

TSNE_SEED       = 42
TSNE_PERPLEXITY = 6    # suitable for ~35 words (perplexity < N/3)
TSNE_ITERS      = 3000

FIG_DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_vectors(wv, groups: dict):
    """
    Extract embedding vectors and metadata for all in-vocabulary words in groups.

    Returns:
        words   – list of word strings
        vectors – np.ndarray of shape (N, embed_dim)
        labels  – list of group-name strings aligned with words
    """
    words, vectors, labels = [], [], []
    skipped = []
    for group_name, word_list in groups.items():
        for word in word_list:
            if word in wv:
                words.append(word)
                vectors.append(wv[word])
                labels.append(group_name)
            else:
                skipped.append(f"{word} ({group_name})")
    if skipped:
        print(f"  [OOV skipped] {', '.join(skipped)}")
    return words, np.array(vectors, dtype=np.float32), labels


def project_pca(vectors: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D via PCA.
    PCA is a linear method that preserves global variance structure.
    The first two principal components capture the largest variance axes
    in the embedding space.
    """
    pca = PCA(n_components=2, random_state=TSNE_SEED)
    return pca.fit_transform(vectors)


def project_tsne(vectors: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D via t-SNE.
    t-SNE is a non-linear method that preserves local neighbourhood structure.
    Perplexity=6 is intentionally low to suit our ~35-word set
    (rule of thumb: perplexity < N/3).
    PCA initialisation provides a more stable starting layout.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERS,
        random_state=TSNE_SEED,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(vectors)


def smart_offsets(coords: np.ndarray):
    """
    Compute per-point text offsets that try to avoid neighbour overlap.
    Points in the upper half get labels above; lower half below.
    Points in the right half get labels to the right; left half to the left.
    This simple heuristic greatly reduces label collision on small word sets.
    """
    cx = coords[:, 0].mean()
    cy = coords[:, 1].mean()
    offsets = []
    for x, y in coords:
        ox = 6 if x >= cx else -6
        oy = 6 if y >= cy else -8
        offsets.append((ox, oy))
    return offsets


def make_legend(group_names, palette):
    """Build legend patch handles."""
    return [
        mpatches.Patch(color=palette[i % len(palette)], label=name)
        for i, name in enumerate(group_names)
    ]


def plot_embedding(
    coords: np.ndarray,
    words: list,
    labels: list,
    group_names: list,
    palette: list,
    title: str,
    ax: plt.Axes,
):
    """
    Scatter plot of 2D word embedding coordinates.
    Words are coloured by semantic group and annotated with smart label offsets
    to reduce overlap. Larger markers are used for improved readability.
    """
    group_index = {name: i for i, name in enumerate(group_names)}
    offsets = smart_offsets(coords)

    for i, (word, label) in enumerate(zip(words, labels)):
        color = palette[group_index[label] % len(palette)]
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=color, s=80, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ox, oy = offsets[i]
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color=color,
            ha="center",
        )

    ax.legend(
        handles=make_legend(group_names, palette),
        loc="upper right",
        fontsize=6.5,
        framealpha=0.7,
        edgecolor="#cccccc",
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Dimension 1", fontsize=8)
    ax.set_ylabel("Dimension 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.25, color="grey")

    # Subtle background
    ax.set_facecolor("#fafafa")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Task-4: Word Embedding Visualisation")
    print("=" * 60)

    # Load best models identified in Task-2
    with open(BEST_MODELS_FILE) as f:
        best_meta = json.load(f)

    arch_models = {}
    for arch, info in best_meta.items():
        print(f"Loading {arch} from {info['model_file']} …")
        arch_models[arch] = Word2Vec.load(info["model_file"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    group_names = list(WORD_GROUPS.keys())

    proj_funcs = {
        "PCA"  : project_pca,
        "tSNE" : project_tsne,
    }

    # ── Individual plots ──────────────────────────────────────────────────────
    # Store coords for reuse in combined figure (avoid recomputing t-SNE twice)
    cached = {}   # (arch, method) → (words, labels, coords)

    for arch, model in arch_models.items():
        words, vectors, labels = collect_vectors(model.wv, WORD_GROUPS)
        print(f"\n{arch}: {len(words)} words available for visualisation")

        for method_name, proj_fn in proj_funcs.items():
            print(f"  Running {method_name} …", end=" ", flush=True)
            coords = proj_fn(vectors)
            cached[(arch, method_name)] = (words, labels, coords)
            print("done")

            fig, ax = plt.subplots(figsize=(9, 7))
            plot_embedding(
                coords, words, labels, group_names, PALETTE,
                title=f"{method_name} Projection — {arch}  "
                      f"(dim={best_meta[arch]['embedding_dim']}, "
                      f"win={best_meta[arch]['window_size']}, "
                      f"neg={best_meta[arch]['negative_samples']})",
                ax=ax,
            )
            plt.tight_layout()
            fname = f"task4_{method_name.lower()}_{arch.lower()}.png"
            path  = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {path}")

    # ── Combined 2×2 figure ───────────────────────────────────────────────────
    # Layout: rows = method (PCA top, t-SNE bottom), cols = arch (CBOW left, SG right)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Word Embedding Projections — CBOW vs Skip-gram  ×  PCA vs t-SNE\n"
        "IIT-Jodhpur Corpus  |  CSL 7640 Assignment 2  |  dim=200, win=2, neg=5",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.patch.set_facecolor("#f5f5f5")

    layout = [
        (0, 0, "CBOW",     "PCA"),
        (0, 1, "SkipGram", "PCA"),
        (1, 0, "CBOW",     "tSNE"),
        (1, 1, "SkipGram", "tSNE"),
    ]
    method_labels = {"PCA": "PCA", "tSNE": "t-SNE"}

    for row, col, arch, method in layout:
        words, labels, coords = cached[(arch, method)]
        plot_embedding(
            coords, words, labels, group_names, PALETTE,
            title=f"{method_labels[method]} — {arch}",
            ax=axes[row][col],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(OUTPUT_DIR, "task4_combined.png")
    fig.savefig(combined_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined 2×2 figure → {combined_path}")

    # ── Print interpretation summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" INTERPRETATION NOTES (see README for full analysis)")
    print("=" * 60)
    print("""
Word Groups (7 categories, ~35 words total):
  Academic Roles | Degree Levels | Academic Activities
  STEM & Tech    | AI/ML Domain  | Course Structure | Places

PCA — what to look for:
  • Linear axes: PC1 often separates 'technical subjects/AI' from
    'administrative/role' vocabulary.
  • CBOW (wide context average) → denser central cluster.
  • Skip-gram (narrow context pairs) → more peripheral spread.

t-SNE — what to look for:
  • Local clusters: Degree Levels (ug/pg/phd/mtech/btech) should group.
  • AI/ML words (machine, neural, algorithm) likely cluster separately
    from administrative words (semester, credit, admission).
  • Places may be isolated — sparse geographic context in corpus.

CBOW vs Skip-gram:
  • CBOW clusters are tighter (avg NN sim 0.9979) — context averaging
    pulls related words together but compresses overall separation.
  • Skip-gram clusters are more spread (0.9743) — individual pair
    training preserves finer distinctions between groups.
""")

    print("Task-4 complete.")


if __name__ == "__main__":
    main()
