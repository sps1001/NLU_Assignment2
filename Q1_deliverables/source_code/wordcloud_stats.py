"""
Task-1: Dataset Statistics + Word Cloud
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Reads the cleaned corpus, prints dataset statistics,
and generates a Word Cloud saved as an image.
"""

import re
import os
from collections import Counter

# pip install wordcloud matplotlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_FILE = "data/corpus.txt"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common English stopwords to exclude from word cloud (they dominate otherwise)
STOPWORDS = {
    "the","a","an","and","or","of","to","in","is","are","was","were",
    "for","on","at","by","with","this","that","these","those","it",
    "be","been","being","have","has","had","will","would","could",
    "should","may","might","shall","do","does","did","not","from",
    "as","its","into","which","also","but","we","our","their","they",
    "he","she","his","her","you","your","all","can","more","than",
    "about","who","one","iitj","ac","www","en","index","php","id",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_corpus(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic-only tokens (removes numbers/symbols)."""
    return re.findall(r"[a-z]{2,}", text.lower())


def print_stats(tokens: list[str], num_docs: int) -> None:
    vocab = set(tokens)
    freq  = Counter(tokens)
    print("\n─── Dataset Statistics ──────────────────────────")
    print(f"  Total documents : {num_docs}")
    print(f"  Total tokens    : {len(tokens):,}")
    print(f"  Vocabulary size : {len(vocab):,} unique words")
    print(f"\n  Top 20 most frequent words:")
    for word, count in freq.most_common(20):
        print(f"    {word:<20} {count:>5}")
    print("─────────────────────────────────────────────────\n")


def make_wordcloud(tokens: list[str], out_path: str) -> None:
    # Filter stopwords for a meaningful cloud
    filtered = [t for t in tokens if t not in STOPWORDS]
    freq = Counter(filtered)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",        # nice colour scheme
        max_words=150,
        collocations=False,        # avoid duplicate bigrams
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words — IITJ Corpus", fontsize=18, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Word cloud saved → {out_path}")


def count_documents(raw_text: str) -> int:
    """Count source blocks (lines starting with ### SOURCE:) as documents.
       Falls back to 1 if the raw corpus has no such markers."""
    count = raw_text.count("### SOURCE:")
    return count if count > 0 else 1


if __name__ == "__main__":
    print(f"Loading corpus from: {CORPUS_FILE}")
    text = load_corpus(CORPUS_FILE)

    # Count docs from raw file if available
    raw_path = "data/raw_corpus.txt"
    num_docs = count_documents(open(raw_path).read()) if os.path.exists(raw_path) else 1

    tokens = tokenize(text)
    print_stats(tokens, num_docs)

    wc_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
    make_wordcloud(tokens, wc_path)
