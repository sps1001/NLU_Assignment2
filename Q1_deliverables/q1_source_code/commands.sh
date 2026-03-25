#!/usr/bin/env bash
# =============================================================================
# commands.sh — Problem 1: Word2Vec Pipeline
# CSL 7640 - Natural Language Understanding, Assignment 2
# Author: B23CS1061
#
# Run this file or copy-paste individual sections to re-run any step.
# All commands assume you are inside problem1_word2vec/ as the working dir.
# Activate the venv first:  source venv310/bin/activate
# =============================================================================

# ── Activate virtualenv ───────────────────────────────────────────────────────
cd "$(dirname "$0")"          # make sure we're in problem1_word2vec/
source venv310/bin/activate

# =============================================================================
# STEP 0 — Scrape IIT Jodhpur website pages  (optional: re-scrape fresh data)
# Writes:  data/raw_corpus.txt  +  data/corpus.txt  (web-only, no PDFs)
# =============================================================================
# python scraper.py

# =============================================================================
# STEP 1 — Build / rebuild full corpus from web scrape + PDFs
# Sources:
#   data/raw_corpus.txt          ← web scrape (scraper.py output)
#   data/docs/*.pdf              ← academic regulation docs
#   data/newsletter/*.pdf        ← IITJ newsletters
#   data/course syllabus/*.pdf   ← B.Tech CSE / AI&DS syllabi
#
# Output:  data/corpus.txt
#   Format: one sentence per line, tokens space-separated, stopwords removed
#
# Stats (current run — 2026-03-25):
#   Sources      : 8  (1 web + 7 PDFs)
#   Sentences    : 3,051
#   Total tokens : 40,197
#   Vocab size   : 5,818 unique words
# =============================================================================
python prepare_corpus.py

# =============================================================================
# STEP 2 — Dataset statistics + Word Cloud
# Reads:   data/corpus.txt
# Writes:  outputs/wordcloud.png  +  prints stats to stdout
# =============================================================================
python wordcloud_stats.py

# =============================================================================
# STEP 3 — Train Word2Vec models (Gensim — 54 models: 27 CBOW + 27 Skip-gram)
# Hyperparameter grid:
#   dim    : 50, 100, 200
#   window : 2, 5, 10
#   neg    : 5, 10, 15
# Reads:   data/corpus.txt
# Writes:  models/*.model  +  outputs/experiment_results.csv
# =============================================================================
python train_word2vec.py

# =============================================================================
# STEP 4 — Train Word2Vec from scratch (PyTorch — CBOW + Skip-gram)
# Implements: negative-sampling loss, unigram^0.75 noise, Adam, Xavier init
# Reads:   data/corpus.txt
# Writes:  outputs/scratch_models/  +  outputs/scratch_loss_curves.png
#          outputs/scratch_vs_gensim.json  +  outputs/scratch_vs_gensim_report.txt
# =============================================================================
python word2vec_scratch.py

# =============================================================================
# STEP 5 — Hyperparameter heatmaps (Task 2 visualizations)
# Reads:   outputs/experiment_results.csv
# Writes:  outputs/task2_heatmap_*.png  +  outputs/task2_heatmaps_combined.png
# =============================================================================
python task2_heatmaps.py

# =============================================================================
# STEP 6 — Semantic analysis: nearest neighbours + analogy experiments (Task 3)
# Uses best models: CBOW(dim=200,win=2,neg=5)  SG(dim=200,win=2,neg=5)
# Probe words : research, student, phd, exam
# Analogies   : ug:btech::pg:?  |  mtech:postgraduate::phd:?
#               professor:teaching::researcher:?  |  semester:student::admission:?
#               undergraduate:degree::phd:?  |  jodhpur:iit::jaipur:?
# Reads:   models/  (best Gensim models from best_models.json)
# Writes:  outputs/task3_semantic_results.txt  +  outputs/task3_semantic_results.json
#
# Results summary (2026-03-25):
#   CBOW  — phd→mtech(0.997), student→registered(0.999), research→faculty(0.999)
#   SG    — phd→mtech(0.991), student→register(0.980),   research→faculty(0.942)
#   Best analogy: SG undergraduate:degree::phd → pg (0.928)
# =============================================================================
python task3_semantic.py

# =============================================================================
# STEP 7 — Embedding visualizations: PCA + t-SNE (Task 4)
# Word groups (35 words, 7 categories):
#   Academic Roles | Degree Levels | Academic Activities | STEM & Tech
#   AI/ML Domain   | Course Structure | Places & Institute
# Projection settings:
#   PCA  — linear, deterministic, seed=42
#   t-SNE — non-linear, perplexity=6, 3000 iters, PCA init, seed=42
# Reads:   best Gensim models (dim=200, win=2, neg=5 for both)
# Writes:  outputs/task4_pca_cbow.png      outputs/task4_pca_skipgram.png
#          outputs/task4_tsne_cbow.png     outputs/task4_tsne_skipgram.png
#          outputs/task4_combined.png      (2x2 grid)
# =============================================================================
python task4_visualize.py

# =============================================================================
# FULL PIPELINE — run everything end-to-end (skip scraper, re-use raw_corpus)
# =============================================================================
# python prepare_corpus.py   && \
# python wordcloud_stats.py  && \
# python train_word2vec.py   && \
# python word2vec_scratch.py && \
# python task2_heatmaps.py   && \
# python task3_semantic.py   && \
# python task4_visualize.py
