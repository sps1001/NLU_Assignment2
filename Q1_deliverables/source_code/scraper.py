"""
IITJ Website Scraper for Word2Vec Corpus Collection
CSL 7640 - Natural Language Understanding, Assignment 2
Author: B23CS1061

Scrapes text content from IIT Jodhpur website pages and saves a clean corpus.
"""

import requests
from bs4 import BeautifulSoup
import re
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# ADD / REMOVE URLs HERE — one per line
# ─────────────────────────────────────────────────────────────────────────────
URLS = [
    # Academic pages
    "https://www.iitj.ac.in/computer-science-engineering/en/undergraduate-programs",
    "https://www.iitj.ac.in/computer-science-engineering/en/postgraduate-programs",
    "https://www.iitj.ac.in/computer-science-engineering/en/programs-for-working-professionals",
    "https://www.iitj.ac.in/computer-science-engineering/en/doctoral-programs",
    # "https://iitj.ac.in/academics/index.php?id=academic_regulations",

    # Department pages
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/chemistry/en/chemistry",
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/electrical-engineering/"

    # Research pages
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive",
    "https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
    "https://www.iitj.ac.in/crf/en/crf",

    # Faculty pages
    "https://www.iitj.ac.in/faculty-positions/en/faculty-positions",
    "https://www.iitj.ac.in/People/List?dept=computer-science-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd"

    # About / Institute
    "https://www.iitj.ac.in/main/en/history",
    "https://www.iitj.ac.in/main/en/director",
   "https://www.iitj.ac.in/main/en/chairman",
    # # Admissions
    # "https://iitj.ac.in/admissions/",
    # "https://iitj.ac.in/admissions/index.php?id=ug_admissions",
    # "https://iitj.ac.in/admissions/index.php?id=pg_admissions",
    # "https://iitj.ac.in/admissions/index.php?id=phd_admissions",

    # # Placements / Student life
    # "https://iitj.ac.in/placement/",
    # "https://iitj.ac.in/student_activity/",
]
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data"
RAW_FILE   = os.path.join(OUTPUT_DIR, "raw_corpus.txt")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "corpus.txt")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# HTML tags whose content should be entirely discarded
DISCARD_TAGS = [
    "script", "style", "noscript",
    "nav", "header", "footer",
    "aside", "form", "button",
    "meta", "link", "head",
    "iframe", "img", "svg",
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def fetch_page(url: str) -> str | None:
    """Download a URL and return raw HTML, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [SKIP] {url}  →  {e}")
        return None


def extract_text(html: str) -> str:
    """
    Parse HTML with BeautifulSoup, strip boilerplate tags,
    and return a single block of clean(ish) English text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove all unwanted structural/non-content tags
    for tag in soup(DISCARD_TAGS):
        tag.decompose()

    # Get visible text, using a space as separator between elements
    text = soup.get_text(separator=" ")
    return text


def clean_text(raw: str) -> str:
    """
    Clean raw extracted text:
      - keep only ASCII (removes Hindi / other scripts)
      - collapse whitespace
      - remove lines that are too short to be real sentences (nav fragments, etc.)
    """
    # Keep only printable ASCII — removes Devanagari and other non-English text
    ascii_text = raw.encode("ascii", errors="ignore").decode("ascii")

    # Lowercase
    lower = ascii_text.lower()

    # Remove URLs
    no_urls = re.sub(r"https?://\S+|www\.\S+", " ", lower)

    # Remove email addresses
    no_email = re.sub(r"\S+@\S+\.\S+", " ", no_urls)

    # Replace newlines / tabs with spaces
    flat = re.sub(r"[\n\r\t]+", " ", no_email)

    # Remove repeated special characters (e.g. "-----", "=====", ".....")
    no_repeated = re.sub(r"([^\w\s])\1{2,}", " ", flat)

    # Remove digits-only tokens (page numbers, phone numbers, etc.)
    no_digits = re.sub(r"\b\d+\b", " ", no_repeated)

    # Collapse multiple spaces into one
    clean = re.sub(r" {2,}", " ", no_digits).strip()

    # Filter out very short tokens / fragments after splitting into sentences
    sentences = re.split(r"[.!?]", clean)
    good_sentences = [
        s.strip() for s in sentences
        if len(s.strip().split()) >= 5  # keep only lines with ≥5 words
    ]

    return ". ".join(good_sentences)


def scrape_all(urls: list[str]) -> list[str]:
    """Scrape all URLs and return a list of cleaned text blocks."""
    results = []
    total = len(urls)

    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{total}] Fetching: {url}")
        html = fetch_page(url)

        if html is None:
            continue  # already printed skip message

        raw  = extract_text(html)
        text = clean_text(raw)

        if len(text.split()) < 20:
            print(f"  [SKIP] Too little text extracted from {url}")
            continue

        results.append(f"### SOURCE: {url}\n{text}")
        print(f"  [OK]   {len(text.split()):,} words extracted")

        time.sleep(1)  # polite delay between requests

    return results


def save(blocks: list[str], path: str) -> None:
    """Write text blocks to a file, one block per source."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
    print(f"\nSaved {len(blocks)} documents → {path}")


def print_stats(path: str) -> None:
    """Print basic corpus statistics."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Strip source markers before counting
    clean = re.sub(r"### SOURCE:.*\n", "", content)
    tokens = clean.split()
    vocab  = set(tokens)

    print("\n─── Corpus Statistics ──────────────────────")
    print(f"  Documents  : {content.count('### SOURCE:')}")
    print(f"  Tokens     : {len(tokens):,}")
    print(f"  Vocabulary : {len(vocab):,} unique words")
    print("────────────────────────────────────────────\n")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" IITJ Corpus Scraper")
    print("=" * 60)

    blocks = scrape_all(URLS)

    if not blocks:
        print("\n[ERROR] No text collected. Check URLs or network connection.")
    else:
        # Save raw (all sources concatenated)
        save(blocks, RAW_FILE)

        # Save a plain version (no source headers) — used as model input
        plain_blocks = [re.sub(r"### SOURCE:.*\n", "", b) for b in blocks]
        save(plain_blocks, CLEAN_FILE)

        print_stats(CLEAN_FILE)
        print(f"Done!\n  Raw corpus  → {RAW_FILE}\n  Clean corpus → {CLEAN_FILE}")
