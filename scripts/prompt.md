# Prompt: Generate the PubMed Cleaning Script

Create a standalone executable Python 3 script named `clean_pubmed.py`. It is a data-cleaning CLI for a PubMed cancer dataset and the Microsoft CancerGUIDE dataset. Use only the Python standard library plus runtime dependencies `datasets`, `pandas`, and `tqdm` (although `tqdm` does not need to be used). The finished script must match all behavior below.

## Module setup

- Begin with `#!/usr/bin/env python3` and a module docstring explaining that it downloads raw Hugging Face datasets, cleans and deduplicates them, and writes per-cancer-type JSONL files for a datagen notebook.
- Import `argparse`, `hashlib`, `html`, `json`, `os`, `re`, `sys`, `unicodedata`, `Counter`, `defaultdict`, and `Path` from `pathlib`. `os` and `Counter` may remain unused.
- Define paths relative to the script:
  - `SCRIPT_DIR = Path(__file__).resolve().parent`
  - `PROJECT_ROOT = SCRIPT_DIR.parent`
  - `DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "source-clean"`
  - `RAW_DIR = PROJECT_ROOT / "data" / "source-raw"` (it may remain unused).
- Define these constants exactly:
  - `MIN_ABSTRACT_CHARS = 200`
  - `MIN_TITLE_CHARS = 10`
  - `MAX_ABSTRACT_CHARS = 15000`
  - `DEDUP_HASH_ALGO = "sha256"` (it may remain unused because the fingerprint helper explicitly calls SHA-256).

## Text helpers

Implement these helpers with the stated names and behavior:

1. `normalize_text(text: str) -> str`
   - Return `""` when the input is falsy or is not a string.
   - Decode HTML entities with `html.unescape`.
   - Normalize Unicode with NFC using `unicodedata.normalize("NFC", text)`.
   - Replace these characters: right and left single quotes with `'`; left and right double quotes with `"`; en dash and em dash with `-`; non-breaking space with a normal space; zero-width space, left-to-right mark, right-to-left mark, and BOM with empty strings.
   - Collapse runs of spaces/tabs to one space using a regex.
   - Collapse three or more newlines to exactly two newlines.
   - Strip leading and trailing whitespace.

2. `is_retracted(title: str, abstract: str) -> bool`
   - Lowercase the joined title and abstract.
   - Return true if it regex-matches any of: `\bretracted\b`, `\bwithdraw[n]?\b`, `\bretraction\b`, `this article has been retracted`, `this paper has been retracted`, `^retracted:`, or `^withdrawn:`.

3. `is_english(text: str) -> bool`
   - Return false for empty input.
   - Collect `isalpha()` characters; return false if fewer than 20 exist.
   - Treat characters with ordinal below `0x0250` as Latin; return true only when more than 85% of alphabetic characters are Latin.

4. `fingerprint(text: str) -> str`
   - Lowercase, remove every character except lowercase ASCII letters, digits, and whitespace using `re.sub(r"[^a-z0-9\s]", "", ...)`.
   - Collapse whitespace, strip it, SHA-256 hash the UTF-8 encoded result, and return the first 16 hexadecimal characters.

## PubMed cleaner

Implement `clean_pubmed_dataset(output_dir: Path, verbose: bool = True) -> dict`.

- Lazily import `pandas as pd` and `load_dataset` from `datasets`. On `ImportError`, print exactly `ERROR: Install required packages: pip install datasets pandas tqdm` and exit with `sys.exit(1)`.
- Initialize this stats object:
  ```python
  {
      "total_raw": 0,
      "per_type_raw": {},
      "per_type_clean": {},
      "dropped": defaultdict(int),
      "dedup_removed": 0,
      "total_clean": 0,
  }
  ```
- When verbose, print a 60-character `=` separator and the step heading `STEP 1: Download PubMed Cancer NLP Dataset`.
- Load exactly `cyberpsych/PubMed-Cancer-NLP-Textual-Dataset` with `trust_remote_code=True`.
- If the returned dataset has a `train` split, convert it to a DataFrame with `.to_pandas()` and, when verbose, print the total record count, column list, and a first-row dictionary (or `empty` if no rows). If no `train` split exists, print available split keys and use an empty `pd.DataFrame()`.
- Detect columns by looping through DataFrame columns and lowercasing/stripping each name:
  - Any name containing `title` becomes the title column.
  - Any name containing `abstract` becomes the abstract column.
  - Any name containing `label`, `cancer`, `type`, or `category` becomes the cancer type column.
  - Later matches may overwrite prior matches.
- Verbosely print the column map. Set a `has_type_col` flag from whether `cancer_type` was found, even if it is unused.
- Print `STEP 2: Clean and Filter` with the same separator when verbose.
- Use one global `seen_fingerprints` set and a `defaultdict(list)` named `records_by_type`.
- Choose title and abstract columns from the detected map, falling back to DataFrame columns 0 and 1 respectively when they exist. If either cannot be identified, print an error containing the DataFrame column list and exit with `sys.exit(1)`.
- Set `stats["total_raw"]` to DataFrame length. Iterate with `df_all.iterrows()`.
- For every row:
  - Coerce the selected title and abstract values with `str(row.get(column, ""))`.
  - If a type column exists and its row value is `pd.notna`, use its lowercased, stripped string with spaces replaced by underscores; otherwise use `unknown`.
  - Remove all non-`a-z`, non-digits, and non-underscores from the cancer type. Replace an empty result with `unknown`.
  - Increment `stats["per_type_raw"][cancer_type]`.
  - Normalize title and abstract with `normalize_text`.
  - Drop and count records in `stats["dropped"]` under these keys, in this precise order: `no_title` if title length is below 10; `short_abstract` if abstract length is below 200; `too_long` if abstract length exceeds 15000; `retracted` if `is_retracted` is true; `non_english` if `is_english` is false; `duplicate` when its abstract fingerprint was seen before.
  - On duplicates, also increment `stats["dedup_removed"]`. Add a fingerprint to the global seen set only after it passes all prior filters.
  - For retained records, make `passage = f"{title}\n\n{abstract}"` and append this dictionary to its cancer-type list:
    ```python
    {
        "id": f"pubmed_{cancer_type}_{idx}",
        "title": title,
        "abstract": abstract,
        "passage": passage,
        "cancer_type": cancer_type,
        "char_count": len(passage),
        "word_count": len(passage.split()),
    }
    ```
- Print `STEP 3: Write Clean Files` when verbose, create `output_dir` with parents, and iterate retained cancer types in sorted order.
- Write each type to `pubmed_<cancer_type>.jsonl`, one `json.dumps(record)` per line. Update `per_type_clean` and `total_clean`. In verbose mode, print each type, count, integer average character count, and output filename.
- Store the total in `stats["total_clean"]`; print total-clean/raw and drop reasons when verbose; then return stats.

## CancerGUIDE cleaner

Implement `clean_cancerguide(output_dir: Path, verbose: bool = True) -> dict`.

- Lazily import `load_dataset`. On import failure, print exactly `ERROR: Install required packages: pip install datasets` and exit.
- Begin stats as `{"structured": 0, "unstructured": 0}`. Verbosely print `STEP 4: Download & Clean CancerGUIDE` with a 60-character separator.
- Ensure `output_dir` exists.
- Process `synthetic_structured` and `synthetic_unstructured` in that order. For each, form a short name by removing `synthetic_`.
- Try loading `microsoft/CancerGUIDE` with the subset name and `trust_remote_code=True`, then use its `train` split. If any exception occurs, print a visible warning with the failed subset and exception when verbose and continue to the next subset.
- For a loaded subset, verbosely display uppercase short name, raw count, and `subset.column_names`.
- Write to `cancerguide_<short_name>.jsonl`. Enumerate the split, normalize `patient_note` and `label`, and use `row.get("patient_id", f"cg_{short_name}_{idx}")` as ID.
- Skip records whose normalized patient note is shorter than 50 characters. Write retained records as JSONL with exactly `id` (converted to string), `patient_note`, `treatment_recommendation` (the normalized label), `subset`, and `char_count` equal to the combined patient-note and label lengths.
- Set the per-subset stats count and print its count and filename when verbose. Return stats.

## CLI and report

- Implement `main()` with `argparse.ArgumentParser(description="Clean PubMed Cancer & CancerGUIDE datasets")`.
- Add `--output-dir` as a `Path`, defaulting to `DEFAULT_OUTPUT_DIR`; `--skip-pubmed` and `--skip-cancerguide` as `store_true`; and `--quiet` as `store_true`.
- Set `verbose = not args.quiet` and initialize `report = {"pubmed": {}, "cancerguide": {}}`.
- When verbose, print the pipeline title and selected output path.
- Unless skipped, call the PubMed cleaner first and CancerGUIDE cleaner second, storing their returned stats under their corresponding report keys.
- Always write `cleaning_report.json` in the output directory. Convert `defaultdict` values into ordinary dictionaries through a JSON serialize/deserialize pass using a `default` lambda before writing the final report with `indent=2`.
- When verbose, print a `CLEANING COMPLETE` section. Include PubMed clean/raw totals unless PubMed was skipped; include CancerGUIDE total plus structured/unstructured counts unless it was skipped; print the report path; and finish with `Next: Run the datagen notebook to generate training data.`
- End with `if __name__ == "__main__": main()`.

Preserve this procedural design and standard-library file I/O. Do not add logging frameworks, progress bars, new CLI flags, retries, schema validation, or behavior not specified above.