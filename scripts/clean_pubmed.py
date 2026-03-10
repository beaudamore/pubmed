#!/usr/bin/env python3
"""
PubMed Cancer & CancerGUIDE — Data Cleaning Pipeline
=====================================================

Downloads raw datasets from HuggingFace, cleans, deduplicates, and writes
per-cancer-type JSONL files to source-clean/ for the datagen notebook.

Usage:
    python scripts/clean_pubmed.py

    # Or with a custom output directory:
    python scripts/clean_pubmed.py --output-dir /path/to/source-clean

Requires:
    pip install datasets pandas tqdm

Outputs:
    source-clean/
    ├── pubmed_bone_cancer.jsonl
    ├── pubmed_brain_tumour.jsonl
    ├── pubmed_breast_cancer.jsonl
    ├── pubmed_colon_cancer.jsonl
    ├── pubmed_gastric_cancer.jsonl
    ├── pubmed_kidney_cancer.jsonl
    ├── pubmed_lung_cancer.jsonl
    ├── pubmed_ovarian_cancer.jsonl
    ├── pubmed_prostate_cancer.jsonl
    ├── pubmed_skin_cancer.jsonl
    ├── cancerguide_structured.jsonl
    ├── cancerguide_unstructured.jsonl
    └── cleaning_report.json
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ── Resolve project paths ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "source-clean"
RAW_DIR = PROJECT_ROOT / "data" / "source-raw"

# ── PubMed CSV filenames → cancer type labels ────────────────────────
PUBMED_FILES = {
    "bone_cancer_samples.csv": "bone_cancer",
    "brain_tumour_samples.csv": "brain_tumour",
    "breast_cancer_samples.csv": "breast_cancer",
    "colon_cancer_samples.csv": "colon_cancer",
    "gastric_cancer_samples.csv": "gastric_cancer",
    "kidney_cancer_samples.csv": "kidney_cancer",
    "lung_cancer_samples.csv": "lung_cancer",
    "ovarian_cancer_samples.csv": "ovarian_cancer",
    "prostate_cancer_samples.csv": "prostate_cancer",
    "skin_cancer_samples.csv": "skin_cancer",
}

# ── Cleaning thresholds ──────────────────────────────────────────────
MIN_ABSTRACT_CHARS = 200       # Drop abstracts shorter than this
MIN_TITLE_CHARS = 10           # Drop entries with no real title
MAX_ABSTRACT_CHARS = 15000     # Cap extremely long entries (rare edge cases)
DEDUP_HASH_ALGO = "sha256"     # For fingerprinting abstracts


def normalize_text(text: str) -> str:
    """Clean a single text field: HTML entities, Unicode, whitespace."""
    if not text or not isinstance(text, str):
        return ""

    # 1. HTML entity decode (PubMed has &amp;, &lt;, &gt;, &#x2019;, etc.)
    text = html.unescape(text)

    # 2. Unicode normalization (NFC — composed form)
    text = unicodedata.normalize("NFC", text)

    # 3. Replace common ligatures and special chars
    replacements = {
        "\u2019": "'",   # right single quote → apostrophe
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\u200e": "",    # left-to-right mark
        "\u200f": "",    # right-to-left mark
        "\ufeff": "",    # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 4. Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_retracted(title: str, abstract: str) -> bool:
    """Check if a paper has been retracted or withdrawn."""
    combined = (title + " " + abstract).lower()
    retract_patterns = [
        r"\bretracted\b",
        r"\bwithdraw[n]?\b",
        r"\bretraction\b",
        r"this article has been retracted",
        r"this paper has been retracted",
        r"^retracted:",
        r"^withdrawn:",
    ]
    return any(re.search(p, combined) for p in retract_patterns)


def is_english(text: str) -> bool:
    """Rough heuristic: check if text is predominantly English/Latin characters.
    
    PubMed is mostly English but some entries have Chinese/Japanese/Korean abstracts.
    We check if >85% of alpha characters are Latin.
    """
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 20:
        return False  # Too short to tell
    latin_chars = sum(1 for c in alpha_chars if ord(c) < 0x0250)  # Basic Latin + Latin Extended
    return (latin_chars / len(alpha_chars)) > 0.85


def fingerprint(text: str) -> str:
    """Create a dedup fingerprint from normalized lowercase text."""
    # Normalize aggressively for dedup: lowercase, strip punctuation, collapse spaces
    clean = re.sub(r"[^a-z0-9\s]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return hashlib.sha256(clean.encode()).hexdigest()[:16]


def clean_pubmed_dataset(output_dir: Path, verbose: bool = True) -> dict:
    """Download, clean, and save PubMed Cancer NLP dataset."""
    try:
        import pandas as pd
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install required packages: pip install datasets pandas tqdm")
        sys.exit(1)

    stats = {
        "total_raw": 0,
        "per_type_raw": {},
        "per_type_clean": {},
        "dropped": defaultdict(int),
        "dedup_removed": 0,
        "total_clean": 0,
    }

    # ── Download dataset ──────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("STEP 1: Download PubMed Cancer NLP Dataset")
        print("=" * 60)

    ds = load_dataset("cyberpsych/PubMed-Cancer-NLP-Textual-Dataset", trust_remote_code=True)

    # The dataset has a single 'train' split with all cancer types
    # But the HF repo actually has 10 separate CSV files
    # Let's check the structure
    if "train" in ds:
        df_all = ds["train"].to_pandas()
        if verbose:
            print(f"  Loaded {len(df_all)} total records")
            print(f"  Columns: {list(df_all.columns)}")
            print(f"  Sample columns values (first row): {dict(df_all.iloc[0]) if len(df_all) > 0 else 'empty'}")
    else:
        print(f"  Available splits: {list(ds.keys())}")
        # Try loading individual files
        df_all = pd.DataFrame()

    # ── Detect column names ───────────────────────────────────────────
    # PubMed CSV structure varies — detect title/abstract/label columns
    cols = [c.lower().strip() for c in df_all.columns]
    col_map = {}

    for c in df_all.columns:
        cl = c.lower().strip()
        if "title" in cl:
            col_map["title"] = c
        elif "abstract" in cl:
            col_map["abstract"] = c
        elif "label" in cl or "cancer" in cl or "type" in cl or "category" in cl:
            col_map["cancer_type"] = c

    if verbose:
        print(f"  Column mapping: {col_map}")

    # Fallback: if no cancer_type column, we'll need to infer from the data
    has_type_col = "cancer_type" in col_map

    # ── Clean each record ─────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 2: Clean and Filter")
        print(f"{'=' * 60}")

    seen_fingerprints = set()
    records_by_type = defaultdict(list)

    title_col = col_map.get("title", df_all.columns[0] if len(df_all.columns) > 0 else None)
    abstract_col = col_map.get("abstract", df_all.columns[1] if len(df_all.columns) > 1 else None)
    type_col = col_map.get("cancer_type", None)

    if title_col is None or abstract_col is None:
        print(f"ERROR: Cannot identify title/abstract columns from: {list(df_all.columns)}")
        sys.exit(1)

    stats["total_raw"] = len(df_all)

    for idx, row in df_all.iterrows():
        raw_title = str(row.get(title_col, ""))
        raw_abstract = str(row.get(abstract_col, ""))

        # Determine cancer type
        if type_col and pd.notna(row.get(type_col)):
            cancer_type = str(row[type_col]).lower().strip().replace(" ", "_")
        else:
            cancer_type = "unknown"

        # Normalize cancer type name
        cancer_type = re.sub(r"[^a-z0-9_]", "", cancer_type)
        if not cancer_type:
            cancer_type = "unknown"

        stats["per_type_raw"][cancer_type] = stats["per_type_raw"].get(cancer_type, 0) + 1

        # ── Apply filters ──
        title = normalize_text(raw_title)
        abstract = normalize_text(raw_abstract)

        # Filter: missing title
        if len(title) < MIN_TITLE_CHARS:
            stats["dropped"]["no_title"] += 1
            continue

        # Filter: missing or too-short abstract
        if len(abstract) < MIN_ABSTRACT_CHARS:
            stats["dropped"]["short_abstract"] += 1
            continue

        # Filter: too long (likely concatenation errors)
        if len(abstract) > MAX_ABSTRACT_CHARS:
            stats["dropped"]["too_long"] += 1
            continue

        # Filter: retracted
        if is_retracted(title, abstract):
            stats["dropped"]["retracted"] += 1
            continue

        # Filter: non-English
        if not is_english(abstract):
            stats["dropped"]["non_english"] += 1
            continue

        # Filter: dedup by abstract fingerprint
        fp = fingerprint(abstract)
        if fp in seen_fingerprints:
            stats["dropped"]["duplicate"] += 1
            stats["dedup_removed"] += 1
            continue
        seen_fingerprints.add(fp)

        # ── Build clean record ──
        # Combine title + abstract into a single passage
        passage = f"{title}\n\n{abstract}"

        record = {
            "id": f"pubmed_{cancer_type}_{idx}",
            "title": title,
            "abstract": abstract,
            "passage": passage,
            "cancer_type": cancer_type,
            "char_count": len(passage),
            "word_count": len(passage.split()),
        }

        records_by_type[cancer_type].append(record)

    # ── Write per-type JSONL files ────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 3: Write Clean Files")
        print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_clean = 0

    for cancer_type, records in sorted(records_by_type.items()):
        outfile = output_dir / f"pubmed_{cancer_type}.jsonl"
        with open(outfile, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        stats["per_type_clean"][cancer_type] = len(records)
        total_clean += len(records)

        if verbose:
            avg_len = sum(r["char_count"] for r in records) // max(len(records), 1)
            print(f"  {cancer_type:25s} {len(records):>6,} records  (avg {avg_len:,} chars)  → {outfile.name}")

    stats["total_clean"] = total_clean

    if verbose:
        print(f"\n  Total clean: {total_clean:,} / {stats['total_raw']:,} raw")
        print(f"  Drop reasons: {dict(stats['dropped'])}")

    return stats


def clean_cancerguide(output_dir: Path, verbose: bool = True) -> dict:
    """Download, clean, and save CancerGUIDE dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install required packages: pip install datasets")
        sys.exit(1)

    stats = {"structured": 0, "unstructured": 0}

    if verbose:
        print(f"\n{'=' * 60}")
        print("STEP 4: Download & Clean CancerGUIDE")
        print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for subset_name in ["synthetic_structured", "synthetic_unstructured"]:
        short_name = subset_name.replace("synthetic_", "")

        try:
            ds = load_dataset("microsoft/CancerGUIDE", subset_name, trust_remote_code=True)
            subset = ds["train"]  # each config has a 'train' split
        except Exception as e:
            if verbose:
                print(f"  ⚠ Subset '{subset_name}' failed to load: {e}")
            continue

        if verbose:
            print(f"\n  [{short_name.upper()}]")
            print(f"    Raw records: {len(subset)}")
            print(f"    Columns: {subset.column_names}")

        outfile = output_dir / f"cancerguide_{short_name}.jsonl"
        count = 0

        with open(outfile, "w") as f:
            for idx, row in enumerate(subset):
                patient_note = normalize_text(str(row.get("patient_note", "")))
                label = normalize_text(str(row.get("label", "")))
                patient_id = row.get("patient_id", f"cg_{short_name}_{idx}")

                if len(patient_note) < 50:
                    continue

                record = {
                    "id": str(patient_id),
                    "patient_note": patient_note,
                    "treatment_recommendation": label,
                    "subset": short_name,
                    "char_count": len(patient_note) + len(label),
                }

                f.write(json.dumps(record) + "\n")
                count += 1

        stats[short_name] = count
        if verbose:
            print(f"    Clean records: {count} → {outfile.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean PubMed Cancer & CancerGUIDE datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for clean JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--skip-pubmed", action="store_true", help="Skip PubMed download/clean")
    parser.add_argument("--skip-cancerguide", action="store_true", help="Skip CancerGUIDE download/clean")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet
    report = {"pubmed": {}, "cancerguide": {}}

    if verbose:
        print(f"\nPubMed Cancer Data Cleaning Pipeline")
        print(f"Output: {args.output_dir}\n")

    # ── PubMed ──
    if not args.skip_pubmed:
        report["pubmed"] = clean_pubmed_dataset(args.output_dir, verbose=verbose)

    # ── CancerGUIDE ──
    if not args.skip_cancerguide:
        report["cancerguide"] = clean_cancerguide(args.output_dir, verbose=verbose)

    # ── Save cleaning report ──
    report_file = args.output_dir / "cleaning_report.json"
    # Convert defaultdicts to regular dicts for JSON serialization
    serializable_report = json.loads(json.dumps(report, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
    with open(report_file, "w") as f:
        json.dump(serializable_report, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print("CLEANING COMPLETE")
        print(f"{'=' * 60}")
        if not args.skip_pubmed:
            pm = report["pubmed"]
            print(f"  PubMed:      {pm.get('total_clean', 0):>6,} clean / {pm.get('total_raw', 0):>6,} raw")
        if not args.skip_cancerguide:
            cg = report["cancerguide"]
            print(f"  CancerGUIDE: {cg.get('structured', 0) + cg.get('unstructured', 0):>6,} records (structured: {cg.get('structured', 0)}, unstructured: {cg.get('unstructured', 0)})")
        print(f"  Report:      {report_file}")
        print(f"\n  Next: Run the datagen notebook to generate training data.")


if __name__ == "__main__":
    main()
