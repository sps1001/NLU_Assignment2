"""
prepare_corpus.py
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Builds an expanded, fully-preprocessed corpus by:
  1. Extracting text from PDF documents in data/docs/, data/newsletter/, data/course syllabus/
  2. Merging with the existing web-scraped text (data/raw_corpus.txt)
  3. Applying preprocessing:
       - Remove boilerplate / formatting artifacts
       - Lowercase
       - Remove non-English / non-ASCII characters
       - Remove URLs, emails, digits-only tokens
       - Remove stopwords (function words like is, are, on, it, you, …)
       - Remove excessive punctuation and very short tokens
       - Tokenize into sentences (one per line) for Word2Vec training
  4. Saving stats and writing the cleaned corpus to data/corpus.txt
"""

import os
import re
import pdfplumber
import nltk
from nltk.corpus import stopwords

# ── ensure NLTK data is present ──────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
RAW_CORPUS       = os.path.join(DATA_DIR, "raw_corpus.txt")
OUTPUT_CORPUS    = os.path.join(DATA_DIR, "corpus.txt")

PDF_DIRS = [
    os.path.join(DATA_DIR, "docs"),
    os.path.join(DATA_DIR, "newsletter"),
    os.path.join(DATA_DIR, "course syllabus"),
]

# ── stopwords ─────────────────────────────────────────────────────────────────
# Standard NLTK English stopwords + a few domain-specific noise words
STOP_WORDS = set(stopwords.words("english")) | {
    # extra noise words common in scraped web / PDF boilerplate
    "page", "download", "click", "please", "note", "also",
    "may", "shall", "would", "could", "etc", "eg", "ie",
    "kb", "mb", "pdf", "file", "read", "view", "visit",
    "www", "http", "https", "com", "org", "ac", "in",
    "iit", "iitj",  # extremely high-frequency but carry little semantic weight
}

# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    """
    Extract all text from a PDF file using pdfplumber.
    Returns a single string with page texts joined by spaces.
    """
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
    return " ".join(texts)


def collect_pdfs(dirs: list) -> dict:
    """Walk each directory, extract text from every PDF found."""
    documents = {}  # filename → raw text
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  [SKIP] Directory not found: {d}")
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(d, fname)
                print(f"  [PDF] Extracting: {fpath}")
                raw = extract_pdf(fpath)
                if len(raw.split()) >= 30:
                    documents[fpath] = raw
                    print(f"        → {len(raw.split()):,} raw words")
                else:
                    print(f"        → [SKIP] Too little text")
    return documents

# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_and_tokenize(raw: str) -> list:
    """
    Full preprocessing pipeline. Returns a list of token-lists (one per sentence).

    Steps:
      1. Keep only ASCII (strips Hindi / other scripts)
      2. Lowercase
      3. Remove URLs
      4. Remove email addresses
      5. Replace newlines / tabs with spaces
      6. Remove repeated special characters (--- === ...)
      7. Remove digits-only tokens
      8. Tokenize into words
      9. Keep only alphabetic tokens (removes stray punctuation)
     10. Remove stopwords
     11. Discard tokens shorter than 2 chars
     12. Group back into sentences (re-split on ". ! ?") for Word2Vec context
    """
    # 1. ASCII only
    text = raw.encode("ascii", errors="ignore").decode("ascii")

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 5. Flatten whitespace
    text = re.sub(r"[\n\r\t]+", " ", text)

    # 6. Remove runs of special characters (e.g. "-----", "=====")
    text = re.sub(r"([^\w\s])\1{2,}", " ", text)

    # 7. Remove standalone digit sequences
    text = re.sub(r"\b\d+\b", " ", text)

    # 8. Tokenize into sentences, then words
    #    Use a simple split on sentence-ending punctuation so Word2Vec
    #    sees proper context windows per sentence.
    sentences = re.split(r"[.!?;]", text)

    result = []
    for sent in sentences:
        # 9. Keep only alphabetic tokens
        tokens = re.findall(r"[a-z]+", sent)

        # 10 & 11. Remove stopwords and very short tokens
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]

        if len(tokens) >= 4:   # discard near-empty sentences
            result.append(tokens)

    return result

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" IITJ Corpus Builder — PDF + Web scrape")
    print("=" * 65)

    all_raw_texts = []   # list of (source_label, raw_string)

    # ── 1. Load existing web-scraped corpus ──────────────────────────────────
    if os.path.exists(RAW_CORPUS):
        with open(RAW_CORPUS, encoding="utf-8") as f:
            web_text = f.read()
        # Strip source markers (### SOURCE: …) before treating as plain text
        web_plain = re.sub(r"### SOURCE:.*\n", "", web_text)
        all_raw_texts.append(("web_scrape", web_plain))
        print(f"\n[WEB]  Loaded raw_corpus.txt  ({len(web_plain.split()):,} raw words)")
    else:
        print(f"\n[WARN] raw_corpus.txt not found — skipping web data")

    # ── 2. Extract PDFs ───────────────────────────────────────────────────────
    print("\n[PDF]  Scanning PDF directories …")
    pdf_docs = collect_pdfs(PDF_DIRS)
    for path, text in pdf_docs.items():
        all_raw_texts.append((os.path.basename(path), text))

    if not all_raw_texts:
        print("\n[ERROR] No text collected. Aborting.")
        return

    # ── 3. Preprocess everything ──────────────────────────────────────────────
    print("\n[PROC] Preprocessing …")
    all_sentences = []       # list of token lists
    doc_count     = len(all_raw_texts)

    for label, raw in all_raw_texts:
        sents = clean_and_tokenize(raw)
        all_sentences.extend(sents)
        print(f"       {label:<45} → {len(sents):,} sentences")

    # ── 4. Write corpus (one sentence per line, tokens space-separated) ───────
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(" ".join(sent) + "\n")

    # ── 5. Stats ──────────────────────────────────────────────────────────────
    all_tokens = [t for sent in all_sentences for t in sent]
    vocab      = set(all_tokens)

    print("\n─── Corpus Statistics ──────────────────────────────────────")
    print(f"  Sources (documents) : {doc_count}")
    print(f"  Sentences           : {len(all_sentences):,}")
    print(f"  Total tokens        : {len(all_tokens):,}")
    print(f"  Vocabulary size     : {len(vocab):,} unique words")
    print(f"  Output file         : {OUTPUT_CORPUS}")
    print("────────────────────────────────────────────────────────────\n")

    print("Done!")


if __name__ == "__main__":
    main()
