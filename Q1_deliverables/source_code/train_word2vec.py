"""
Task-2: Word2Vec Model Training (CBOW + Skip-gram)
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Trains multiple Word2Vec models with different hyperparameter combinations
and saves results + models for downstream tasks (Task-3, Task-4).
"""

import re
import os
import json
import itertools
from time import time

import gensim
from gensim.models import Word2Vec
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_FILE = "data/corpus.txt"
MODELS_DIR  = "models"
OUTPUT_DIR  = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameter grid — edit freely
EMBEDDING_DIMS   = [50, 100, 200]        # vector size
WINDOW_SIZES     = [2, 5, 10]            # context window
NEGATIVE_SAMPLES = [5, 10, 15]           # negative samples (only for sg)

EPOCHS   = 10
MIN_COUNT = 2   # ignore words that appear fewer than this many times
WORKERS  = 4    # parallel threads
# ─────────────────────────────────────────────────────────────────────────────


def load_sentences(path: str) -> list[list[str]]:
    """
    Read the cleaned corpus and return a list of sentences.
    Each sentence is a list of lowercase alphabetic tokens.

    Supports both formats:
      - One sentence per line (prepare_corpus.py output — tokens space-separated)
      - Period-delimited single-block text (legacy scraper.py output)
    The function detects the format automatically by checking whether the file
    has many lines or a single/few long lines.
    """
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip() for l in f if l.strip()]

    sentences = []

    if len(lines) > 100:
        # New format: one pre-tokenised sentence per line
        for line in lines:
            tokens = re.findall(r"[a-z]{2,}", line.lower())
            if len(tokens) >= 3:
                sentences.append(tokens)
    else:
        # Legacy format: whole corpus in a few lines, split on sentence punctuation
        text = " ".join(lines)
        for sent in re.split(r"[.!?]", text):
            tokens = re.findall(r"[a-z]{2,}", sent.lower())
            if len(tokens) >= 3:
                sentences.append(tokens)

    print(f"Loaded {len(sentences):,} sentences from corpus.")
    return sentences


def model_name(arch: str, dim: int, win: int, neg: int) -> str:
    """Unique identifier for a model configuration."""
    return f"{arch}_dim{dim}_win{win}_neg{neg}"


def train_model(
    sentences: list[list[str]],
    sg: int,          # 0 = CBOW, 1 = Skip-gram
    dim: int,
    window: int,
    negative: int,
) -> tuple[Word2Vec, float]:
    """Train one Word2Vec model and return (model, training_time_seconds)."""
    t0 = time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=MIN_COUNT,
        sg=sg,
        negative=negative,
        epochs=EPOCHS,
        workers=WORKERS,
        seed=42,
    )
    elapsed = time() - t0
    return model, elapsed


def intrinsic_score(model: Word2Vec) -> float:
    """
    Simple intrinsic quality proxy: average cosine similarity of each word
    to its single nearest neighbour. Higher = tighter clusters = better
    semantic structure (rough heuristic, not a gold-standard benchmark).
    """
    vocab = list(model.wv.index_to_key)
    sample = vocab[:200]               # use top-200 frequent words
    total = 0.0
    for word in sample:
        sims = model.wv.most_similar(word, topn=1)
        total += sims[0][1]
    return round(total / len(sample), 4)


def run_experiments(sentences: list[list[str]]) -> pd.DataFrame:
    """
    Train all CBOW + Skip-gram combinations and collect results into a DataFrame.
    """
    records = []
    combos  = list(itertools.product(EMBEDDING_DIMS, WINDOW_SIZES, NEGATIVE_SAMPLES))
    total   = len(combos) * 2          # × 2 architectures
    idx     = 0

    for arch_name, sg in [("CBOW", 0), ("SkipGram", 1)]:
        for dim, win, neg in combos:
            idx += 1
            name = model_name(arch_name, dim, win, neg)
            print(f"[{idx}/{total}] Training {name} ...", end=" ", flush=True)

            model, elapsed = train_model(sentences, sg, dim, win, neg)
            score = intrinsic_score(model)

            # Save model to disk for Task-3 / Task-4
            model_path = os.path.join(MODELS_DIR, f"{name}.model")
            model.save(model_path)

            print(f"done in {elapsed:.1f}s  |  avg_sim={score}")

            records.append({
                "Architecture"    : arch_name,
                "Embedding Dim"   : dim,
                "Window Size"     : win,
                "Negative Samples": neg,
                "Vocab Size"      : len(model.wv),
                "Train Time (s)"  : round(elapsed, 2),
                "Avg NN Sim"      : score,   # intrinsic quality proxy
                "Model File"      : model_path,
            })

    return pd.DataFrame(records)


def print_results_table(df: pd.DataFrame) -> None:
    display_cols = [
        "Architecture", "Embedding Dim", "Window Size",
        "Negative Samples", "Vocab Size", "Train Time (s)", "Avg NN Sim"
    ]
    print("\n─── Experiment Results ──────────────────────────────────────────────")
    print(df[display_cols].to_string(index=False))
    print("─────────────────────────────────────────────────────────────────────\n")


def save_best_models(df: pd.DataFrame) -> None:
    """
    Identify the best CBOW and best Skip-gram model (by Avg NN Sim)
    and copy their paths to a JSON file for easy loading in Task-3/4.
    """
    best = {}
    for arch in ["CBOW", "SkipGram"]:
        row = df[df["Architecture"] == arch].sort_values("Avg NN Sim", ascending=False).iloc[0]
        best[arch] = {
            "model_file"      : row["Model File"],
            "embedding_dim"   : int(row["Embedding Dim"]),
            "window_size"     : int(row["Window Size"]),
            "negative_samples": int(row["Negative Samples"]),
            "avg_nn_sim"      : float(row["Avg NN Sim"]),
        }
        print(f"Best {arch}: dim={row['Embedding Dim']}, "
              f"win={row['Window Size']}, neg={row['Negative Samples']}, "
              f"avg_sim={row['Avg NN Sim']}")

    out = os.path.join(OUTPUT_DIR, "best_models.json")
    with open(out, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest model paths saved → {out}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Task-2: Word2Vec Training")
    print(f" gensim version: {gensim.__version__}")
    print("=" * 60)

    sentences = load_sentences(CORPUS_FILE)

    df = run_experiments(sentences)

    # Save full results table as CSV (useful for the report)
    csv_path = os.path.join(OUTPUT_DIR, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved → {csv_path}")

    print_results_table(df)
    save_best_models(df)

    print("\nAll models saved in:", MODELS_DIR)
    print("Next → run  task3_semantic.py  for nearest neighbours + analogies")
