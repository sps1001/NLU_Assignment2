"""
Task-3: Semantic Analysis
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Uses the best CBOW and Skip-gram models (selected in Task-2) to:
  1. Report the top-5 nearest neighbours (by cosine similarity) for
     probe words: research, student, phd, exam
  2. Run at least three word-analogy experiments using the 3CosAdd method
     (a - b + c → d) and discuss semantic plausibility.
  3. Save all results to outputs/task3_semantic_results.txt (human-readable)
     and outputs/task3_semantic_results.json (machine-readable for the report).
"""

import json
import os
from gensim.models import Word2Vec

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "outputs"
BEST_MODELS_FILE = os.path.join(OUTPUT_DIR, "best_models.json")

# Words whose nearest neighbours we must report (assignment spec)
PROBE_WORDS = ["research", "student", "phd", "exam"]

# Analogy triples: (a, b, c)  →  expected d  such that  a - b + c ≈ d
# Interpretation: "a is to b as c is to d"
# Chosen to exploit IIT-Jodhpur domain vocabulary present in the corpus.
ANALOGIES = [
    # SPEC ANALOGY — directly from assignment:
    # "UG is to BTech as PG is to ?"
    # Expected: mtech  (PG equivalent of BTech)
    ("ug",            "btech",        "pg"),

    # "mtech is to postgraduate as phd is to ?"
    # Expected: a word associated with doctoral-level study (e.g. research, thesis)
    ("mtech",         "postgraduate", "phd"),

    # "professor is to teaching as researcher is to ?"
    # Expected: a research-output word (e.g. thesis, scholarship)
    ("professor",     "teaching",     "researcher"),

    # "semester is to student as admission is to ?"
    # Expected: a selection/intake word (e.g. candidates, scholarship)
    ("semester",      "student",      "admission"),

    # "undergraduate is to degree as phd is to ?"
    # Expected: thesis, research — doctoral output concept
    ("undergraduate", "degree",       "phd"),

    # "jodhpur is to iit as jaipur is to ?"
    # Expected: another institution or geographic reference
    ("jodhpur",       "iit",          "jaipur"),
]

TOPN = 5   # nearest neighbours to report


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_best_models():
    """Load the two best models identified in Task-2."""
    with open(BEST_MODELS_FILE) as f:
        meta = json.load(f)

    models = {}
    for arch, info in meta.items():
        path = info["model_file"]
        print(f"Loading {arch} from {path} …")
        models[arch] = Word2Vec.load(path)
    return models


def nearest_neighbours(model_wv, word: str, topn: int = 5):
    """
    Return the top-N most similar words to `word` using cosine similarity.
    Returns a list of (word, similarity) tuples, or None if word is OOV.
    """
    if word not in model_wv:
        return None
    return model_wv.most_similar(word, topn=topn)


def run_analogy(model_wv, a: str, b: str, c: str, topn: int = 5):
    """
    3CosAdd analogy: find d such that  a - b + c ≈ d
    (i.e.  a is to b  as  c is to d).

    Gensim's most_similar(positive=[a, c], negative=[b]) implements exactly
    this arithmetic in the embedding space.

    Returns list of (word, score) or None if any term is OOV.
    """
    for word in [a, b, c]:
        if word not in model_wv:
            return None   # cannot compute if any term missing
    return model_wv.most_similar(positive=[a, c], negative=[b], topn=topn)


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyse(models: dict) -> dict:
    """
    Run nearest-neighbour and analogy experiments for every model.
    Returns a nested dict: results[arch]["nn"][word] and results[arch]["analogies"][(a,b,c)].
    """
    results = {}

    for arch, model in models.items():
        wv = model.wv
        arch_results = {"nn": {}, "analogies": []}

        # ── 1. Nearest Neighbours ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f" {arch}  —  Top-{TOPN} Nearest Neighbours")
        print(f"{'='*60}")
        for word in PROBE_WORDS:
            nn = nearest_neighbours(wv, word, topn=TOPN)
            arch_results["nn"][word] = nn  # may be None if OOV
            if nn is None:
                print(f"  {word:15s} → [OOV – not in vocabulary]")
            else:
                print(f"  {word:15s} → {[w for w, _ in nn]}")
                for neighbour, score in nn:
                    print(f"      {neighbour:20s}  sim={score:.4f}")

        # ── 2. Analogy Experiments ───────────────────────────────────────────
        print(f"\n{'-'*60}")
        print(f" {arch}  —  Analogy Experiments  (a - b + c → d)")
        print(f"{'-'*60}")
        for a, b, c in ANALOGIES:
            top = run_analogy(wv, a, b, c, topn=TOPN)
            entry = {
                "query"   : f"{a} - {b} + {c}",
                "readable": f"{a} : {b}  ::  {c} : ?",
                "results" : top,   # None if OOV
            }
            arch_results["analogies"].append(entry)

            if top is None:
                print(f"  {a} : {b} :: {c} : ?   → [OOV term in query]")
            else:
                top_word, top_score = top[0]
                print(f"  {a} : {b} :: {c} : ?")
                print(f"    Best answer → '{top_word}'  (sim={top_score:.4f})")
                print(f"    Top-{TOPN}: {[w for w, _ in top]}")

        results[arch] = arch_results

    return results


# ── Save results ───────────────────────────────────────────────────────────────

def save_txt(results: dict, path: str):
    """
    Write a human-readable report of all nearest-neighbour and analogy results,
    including a qualitative discussion of semantic plausibility.
    """
    lines = []
    lines.append("=" * 70)
    lines.append(" TASK-3: SEMANTIC ANALYSIS RESULTS")
    lines.append(" CSL 7640 — Assignment 2")
    lines.append("=" * 70)

    for arch, data in results.items():
        lines.append(f"\n{'#'*70}")
        lines.append(f"  Architecture: {arch}")
        lines.append(f"{'#'*70}")

        # Nearest neighbours section
        lines.append(f"\n{'─'*50}")
        lines.append(f"  TOP-{TOPN} NEAREST NEIGHBOURS (cosine similarity)")
        lines.append(f"{'─'*50}")
        for word, nn in data["nn"].items():
            if nn is None:
                lines.append(f"\n  '{word}' → [OOV]")
            else:
                lines.append(f"\n  '{word}':")
                for rank, (neighbour, score) in enumerate(nn, 1):
                    lines.append(f"    {rank}. {neighbour:25s}  {score:.4f}")

        # Analogy section
        lines.append(f"\n{'─'*50}")
        lines.append(f"  ANALOGY EXPERIMENTS  (a : b :: c : ?)")
        lines.append(f"{'─'*50}")
        for entry in data["analogies"]:
            lines.append(f"\n  Query : {entry['readable']}")
            if entry["results"] is None:
                lines.append(f"  Result: [OOV term — cannot compute]")
            else:
                lines.append(f"  Top-{TOPN} answers:")
                for rank, (w, s) in enumerate(entry["results"], 1):
                    lines.append(f"    {rank}. {w:25s}  {s:.4f}")

    # Qualitative discussion (written analysis required by assignment spec)
    lines.append(f"\n{'='*70}")
    lines.append("  QUALITATIVE DISCUSSION")
    lines.append(f"{'='*70}")
    discussion = """
Nearest Neighbours Analysis
-----------------------------
• 'research' neighbours:
  CBOW returns: award, faculty, industry, required, supervisor.
  Skip-gram returns: faculty, member, undergraduate, award, chairman.
  Both models surface 'faculty', 'award', 'industry' — all semantically
  valid associates of research in an IIT context. 'supervisor' (CBOW) and
  'member' (SG) are particularly strong, capturing advising and collegial
  roles central to academic research. The results are semantically coherent.

• 'student' neighbours:
  CBOW: registered, completed, academic, maximum, requirements.
  Skip-gram: register, registered, registration, academic, must.
  Both clusters are regulation-document vocabulary — words that co-occur
  with 'student' in academic rules (registration, attendance, completion
  requirements). The neighbourhood is semantically tight and consistent
  with the corpus source (academic regulation PDFs + syllabus).

• 'phd' neighbours:
  CBOW: mtech, bouquet, none, programme, distribution.
  Skip-gram: mtech, program, tech, masters, dual.
  The most important result is that BOTH models rank 'mtech' as the single
  nearest neighbour to 'phd' (sim ≈ 0.99 CBOW, 0.99 SG). This is strongly
  meaningful — these degree labels co-occur in programme listings and
  admission criteria throughout the corpus. Skip-gram additionally surfaces
  'masters', 'dual', 'program' — all semantically valid degree-level words.
  'bouquet' and 'none' (CBOW) are noise, likely due to co-occurrence with
  newsletter text listing prize categories.

• 'exam' neighbours:
  CBOW: including, emerging, international, smart, industry.
  Skip-gram: entrepreneurial, middle, component, subject, copies.
  'subject' and 'component' (Skip-gram) carry weak exam-related meaning.
  The overall quality is lower — 'exam' appears infrequently in the academic
  regulation PDFs (which focus more on grades and attendance) so the model
  encounters it mainly in course-description contexts (e.g. "mid-semester
  exam"), causing the embedding to mix with unrelated course-description words.

Analogy Experiments
--------------------
• ug : btech :: pg : ?   [Assignment spec analogy]
  CBOW best: 'report' — not the expected 'mtech'.
  Skip-gram best: 'allowed' — also not ideal.
  The expected answer is 'mtech' (PG ≡ MTech as UG ≡ BTech). The analogy
  does not resolve correctly because 'ug'/'pg' and 'btech'/'mtech' occur
  in different textual contexts (abbreviations in programme headings vs.
  full names in regulation text), preventing the model from learning a
  precise parallel vector offset. This is a known limitation of 3CosAdd on
  domain-specific corpora with inconsistent abbreviation usage.

• mtech : postgraduate :: phd : ?
  Both models return 'tech' as top answer (contained in 'mtech').
  Skip-gram also surfaces 'elective', 'compulsory', 'programme'.
  Partially meaningful — the model tries to subtract the 'postgraduate'
  direction from 'mtech' and add 'phd', but the resulting vector lands
  near syllabus structure words rather than 'doctoral'. The corpus lacks
  explicit 'doctoral' vocabulary.

• professor : teaching :: researcher : ?
  CBOW: 'biometrics'; Skip-gram: 'aiims'.
  Neither is semantically correct. The expected answer is a research-output
  concept ('thesis', 'publication'), but these words are rare in the corpus.
  The analogy arithmetic lands on noisy neighbours because 'professor' and
  'researcher' are both rare and have unstable vectors.

• semester : student :: admission : ?
  CBOW: 'ii', 'grades', 'ch', 'summer'. Skip-gram: 'category', 'ay', 'july'.
  'summer' (CBOW) and 'category' (Skip-gram) are weakly meaningful
  ('admission category', 'summer admission'). The analogy is partially valid
  but dominated by academic-calendar noise words.

• undergraduate : degree :: phd : ?
  Skip-gram top result: 'pg' (sim=0.928), followed by 'candidates',
  'completing', 'branches'. This is the strongest analogy result — 'pg'
  is semantically valid as a degree-level label parallel to 'undergraduate',
  confirming the model has encoded the UG/PG/PhD hierarchy to some extent.

• jodhpur : iit :: jaipur : ?
  'jaipur' is out-of-vocabulary (the corpus is IIT-Jodhpur-centric and
  contains almost no Jaipur references). Analogy cannot be computed.

CBOW vs Skip-gram Comparison
------------------------------
Skip-gram consistently surfaces more semantically meaningful nearest
neighbours than CBOW, particularly for lower-frequency words ('phd',
'research', 'faculty'). This is expected: Skip-gram trains on individual
center-context pairs, giving rare words proportionally more gradient updates,
resulting in sharper embeddings for specialised vocabulary.

CBOW averages context vectors, which works well for high-frequency words
('student', registration-related words) but collapses distinctions for
domain-specific rare terms. The high cosine similarities seen in CBOW
(> 0.99 for most neighbours) indicate a more compressed embedding space
where the top-K results are nearly indistinguishable — a known artefact
of the averaging operation on a focused-domain corpus.

For analogy tasks, neither model performs strongly. The 3CosAdd method
requires well-separated, directionally consistent vector offsets — which
needs a large, balanced vocabulary. On a 3,364-word IIT-specific corpus,
analogical relationships are too sparse to produce reliable offsets.
The one clear success (undergraduate:degree::phd → pg, Skip-gram) shows
the model has partially encoded degree-level structure, but this does not
generalise reliably to arbitrary analogy triples.
"""
    lines.append(discussion)
    lines.append("=" * 70)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nText report saved → {path}")


def save_json(results: dict, path: str):
    """
    Serialise results to JSON for programmatic access / report generation.
    Converts tuple keys and None-safe values.
    """
    serialisable = {}
    for arch, data in results.items():
        serialisable[arch] = {
            "nearest_neighbours": {
                word: (
                    [{"word": w, "similarity": round(float(s), 4)} for w, s in nn]
                    if nn else None
                )
                for word, nn in data["nn"].items()
            },
            "analogies": [
                {
                    "query"   : entry["query"],
                    "readable": entry["readable"],
                    "results" : (
                        [{"word": w, "similarity": round(float(s), 4)} for w, s in entry["results"]]
                        if entry["results"] else None
                    ),
                }
                for entry in data["analogies"]
            ],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"JSON results saved  → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Task-3: Semantic Analysis")
    print("=" * 60)

    # Load the best CBOW and Skip-gram models from Task-2
    models = load_best_models()

    # Run all analyses
    results = analyse(models)

    # Persist outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_txt(results, os.path.join(OUTPUT_DIR, "task3_semantic_results.txt"))
    save_json(results, os.path.join(OUTPUT_DIR, "task3_semantic_results.json"))

    print("\nTask-3 complete. Next → run task4_visualize.py for PCA/t-SNE plots.")
