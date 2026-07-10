#!/usr/bin/env python3
"""Build tool-calling augmentation files from the existing PubMed datagen output.

This script is intentionally local and deterministic. It does not call OpenRouter,
PubMed, HuggingFace, or any other network service. The goal is to reuse the costly
SFT/DPO data that already exists, then add a small, explicit layer that teaches the
model the missing behavior:

    biomedical evidence question -> call deep_research_pubmed -> answer from result

The user renamed the original training data directory to:

    data/training-data-backups-before-tool-calling-augmentation

This script treats that backup as read-only source material and recreates:

    data/training-data

It also writes additive files at the top level of data/training-data:

    pubmed_oncologist_v2_tool_sft_messages.jsonl
    pubmed_oncologist_v2_tool_dpo_messages.jsonl
    tool_calling_augmentation_manifest.json

Why message-format JSONL instead of the older ShareGPT `from/value` shape?
OpenAI/Qwen-style tool calling is represented by assistant messages with a
`tool_calls` array and subsequent `role: tool` messages. Flattening that into a
plain assistant text response is exactly how tool-calling behavior gets weakened.

The supplemental tool-SFT notebook should train on the `messages` file with the
model/tokenizer chat template that supports tools. The DPO notebook should use a
tool-aware DPO file or be updated to consume `chosen`/`rejected` message arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BACKUP_DIR = DATA_DIR / "training-data-backups-before-tool-calling-augmentation"
DEFAULT_OUTPUT_DIR = DATA_DIR / "training-data"
DATASET_NAME = "pubmed_oncologist_v2"
DEFAULT_SFT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "loras" / "pubmed_qwen3-14b-sft_training.ipynb"
DEFAULT_DPO_NOTEBOOK = PROJECT_ROOT / "notebooks" / "loras" / "pubmed_dpo_training_v2.ipynb"

ORIGINAL_SFT_FILENAME = f"{DATASET_NAME}.jsonl"
ORIGINAL_DPO_FILENAME = f"{DATASET_NAME}_dpo.jsonl"
TOOL_SFT_FILENAME = f"{DATASET_NAME}_tool_sft_messages.jsonl"
TOOL_DPO_FILENAME = f"{DATASET_NAME}_tool_dpo_messages.jsonl"
MANIFEST_FILENAME = "tool_calling_augmentation_manifest.json"

TOOL_NAME = "deep_research_pubmed"
TOOL_DESCRIPTION = (
    "Search PubMed for biomedical literature, archive new articles into the "
    "OpenWebUI knowledge base, and return article details including PMID, title, "
    "authors, DOI, abstract, entities, keywords, and storage summary. Use this "
    "before answering biomedical literature, oncology, clinical evidence, trial, "
    "biomarker, mechanism, treatment, guideline, or PubMed-style questions."
)

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The biomedical literature search query to run in PubMed.",
                }
            },
            "required": ["query"],
        },
    },
}

TOOL_ROUTING_SYSTEM_APPENDIX = f"""

TOOL USE POLICY:
- You have access to the `{TOOL_NAME}` tool.
- For biomedical literature, oncology, clinical evidence, trial, biomarker,
  mechanism, treatment, guideline, or PubMed-style questions, call `{TOOL_NAME}`
  before answering.
- When you call `{TOOL_NAME}`, do not answer in the same assistant turn. Wait for
  the tool result, then synthesize the answer from the returned evidence.
- Do not invent PMIDs, trial names, statistics, guideline statements, or article
  details. If the tool result is incomplete, say what remains uncertain.
""".rstrip()


@dataclass(frozen=True)
class ToolExample:
    """One derived tool-calling training example."""

    source_index: int
    source_file: str
    cancer_type: str
    question: str
    answer: str
    system_prompt: str
    query: str
    tool_result: str
    call_id: str


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as compact UTF-8 JSONL and return the row count."""
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def count_jsonl_rows(path: Path) -> int:
    """Count non-blank JSONL rows without loading the file into memory."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_notebook_int_constant(path: Path, name: str) -> int | None:
    """Read a simple integer assignment from a notebook source cell.

    The LoRA notebooks are JSON, but their code appears as source strings. This
    intentionally supports only literal forms used by the project configs, such
    as `DPO_MAX_PAIRS = 9000`, `SFT_MAX_EXAMPLES = 3000`, `0`, or `None`.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\b{name}\s*=\s*(None|0|[1-9][0-9]*)", text)
    if not match:
        return None
    value = match.group(1)
    if value in {"None", "0"}:
        return None
    return int(value)


def selected_count(total_rows: int, cap: int | None) -> int:
    """Return the number of rows the notebook would train on after its cap."""
    if cap is None:
        return total_rows
    return min(total_rows, cap)


def resolve_tool_example_limit(
    source_dir: Path,
    max_tool_examples: int | None,
    tool_ratio_to_chosen: float,
    sft_notebook: Path,
    dpo_notebook: Path,
) -> tuple[int, dict[str, Any]]:
    """Choose tool-call row count from the same sample limits as the notebooks.

    By default, tool-call examples track the selected DPO pair count. For the
    active Qwen3-14B notebook this reads `DPO_MAX_PAIRS = 9000`, so the script
    emits 9,000 tool-SFT rows and 9,000 tool-DPO rows unless overridden.
    """
    if tool_ratio_to_chosen <= 0:
        raise ValueError(f"--tool-ratio-to-chosen must be positive, got {tool_ratio_to_chosen}")

    sft_total = count_jsonl_rows(source_dir / ORIGINAL_SFT_FILENAME)
    dpo_total = count_jsonl_rows(source_dir / ORIGINAL_DPO_FILENAME)
    sft_cap = read_notebook_int_constant(sft_notebook, "SFT_MAX_EXAMPLES")
    dpo_cap = read_notebook_int_constant(dpo_notebook, "DPO_MAX_PAIRS")
    sft_selected = selected_count(sft_total, sft_cap)
    dpo_selected = selected_count(dpo_total, dpo_cap)

    if max_tool_examples is None:
        requested = round(dpo_selected * tool_ratio_to_chosen)
        reason = "auto_scaled_from_dpo_selected_pairs"
    else:
        if max_tool_examples <= 0:
            raise ValueError(f"--max-tool-examples must be positive when provided, got {max_tool_examples}")
        requested = max_tool_examples
        reason = "explicit_max_tool_examples"

    resolved = max(1, min(requested, sft_selected))
    details = {
        "limit_reason": reason,
        "tool_ratio_to_chosen": tool_ratio_to_chosen,
        "sft_notebook": str(sft_notebook),
        "dpo_notebook": str(dpo_notebook),
        "sft_total_rows": sft_total,
        "sft_max_examples": sft_cap,
        "sft_selected_rows": sft_selected,
        "dpo_total_rows": dpo_total,
        "dpo_max_pairs": dpo_cap,
        "dpo_selected_pairs": dpo_selected,
        "requested_tool_examples": requested,
        "resolved_tool_examples": resolved,
    }
    return resolved, details


def stable_id(*parts: str, prefix: str = "call") -> str:
    """Create a deterministic OpenAI-style tool call id."""
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def strip_think(text: str) -> str:
    """Remove Qwen-style thinking blocks from text used in synthetic tool output."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    return text.strip()


def first_sentence_window(text: str, max_chars: int = 1400) -> str:
    """Return a readable excerpt without cutting absurdly long tool-result text."""
    clean = re.sub(r"\s+", " ", strip_think(text)).strip()
    if len(clean) <= max_chars:
        return clean
    cutoff = clean.rfind(". ", 0, max_chars)
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return clean[:cutoff].rstrip() + "..."


def normalize_cancer_type(value: str | None) -> str:
    """Convert pubmed_breast_cancer into Breast Cancer for display text."""
    if not value:
        return "Oncology"
    value = value.removeprefix("pubmed_").replace("_", " ").strip()
    return value.title() if value else "Oncology"


def extract_conversation_parts(row: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return (system, user, assistant) from either ShareGPT or helper fields.

    The expensive datagen files are not perfectly uniform. Some rows have explicit
    `question` and `answer` fields; others only have `conversations`. This helper
    keeps the augmentation resilient without changing source data.
    """
    system_prompt = ""
    user_text = row.get("question") or ""
    assistant_text = row.get("answer") or ""

    conversations = row.get("conversations") or []
    for message in conversations:
        role = message.get("from") or message.get("role")
        content = message.get("value") or message.get("content") or ""
        if role == "system" and not system_prompt:
            system_prompt = content
        elif role in {"human", "user"} and not user_text:
            user_text = content
        elif role in {"gpt", "assistant"} and not assistant_text:
            assistant_text = content

    if not user_text or not assistant_text:
        return None
    return system_prompt, user_text, assistant_text


def build_pubmed_query(question: str, cancer_type: str) -> str:
    """Build a compact deterministic PubMed query from the existing question.

    This is not meant to be a perfect search optimizer. It teaches the model to
    put the user's biomedical question into the tool's `query` parameter. The live
    tool already has query variation/spell-check logic for follow-up expansion.
    """
    question = re.sub(r"\s+", " ", question).strip()
    question = re.sub(r"[?]+$", "", question).strip()
    cancer_label = normalize_cancer_type(cancer_type)
    if cancer_label.lower() not in question.lower():
        question = f"{question} {cancer_label}"
    return question[:350]


def build_tool_result(example_index: int, question: str, answer: str, cancer_type: str) -> str:
    """Create a PubMed-tool-shaped result from existing grounded answer text.

    Many validated QA rows no longer carry the original PMID/title/abstract. Rather
    than invent article identifiers, the synthetic result is explicit about being a
    training snapshot derived from the archived PubMed QA corpus. This avoids
    teaching the model to fabricate PMIDs while still teaching the tool-result ->
    final-answer transition.
    """
    excerpt = first_sentence_window(answer)
    cancer_label = normalize_cancer_type(cancer_type)
    pseudo_pmid = f"TRAINING-SNAPSHOT-{example_index:06d}"
    return "\n".join(
        [
            f"🔬 **PubMed Deep Research Results** — {question}",
            "",
            "**Training Snapshot Notice**: This tool result was synthesized from the existing validated PubMed oncology training corpus. It is for tool-calling alignment only and does not assert a real PMID unless one is shown explicitly.",
            "",
            "--- **Article Details** ---",
            f"Search Query: {question}",
            "Retrieved At (UTC): synthetic-training-snapshot",
            "Total Articles Found: 1",
            "Articles in Report: 1 (of 1)",
            "",
            "Article 1",
            f"Title: Existing validated PubMed oncology evidence snapshot for {cancer_label}",
            "Authors: N/A",
            "DOI: N/A",
            f"PMID: {pseudo_pmid}",
            f"Abstract: {excerpt}",
            f"Entities: {cancer_label}",
            "",
        ]
    )


def make_tool_sft_row(example: ToolExample) -> dict[str, Any]:
    """Create one native message-format tool-calling SFT row."""
    system_prompt = (example.system_prompt or "You are a clinical oncologist.").rstrip()
    system_prompt = f"{system_prompt}\n{TOOL_ROUTING_SYSTEM_APPENDIX}"
    arguments = json.dumps({"query": example.query}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.question},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": example.call_id,
                        "type": "function",
                        "function": {"name": TOOL_NAME, "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": example.call_id,
                "name": TOOL_NAME,
                "content": example.tool_result,
            },
            {"role": "assistant", "content": example.answer},
        ],
        "tools": [TOOL_SPEC],
        "source": "tool_calling_augmentation",
        "source_file": example.source_file,
        "source_index": example.source_index,
        "cancer_type": example.cancer_type,
    }


def make_tool_dpo_row(example: ToolExample) -> dict[str, Any]:
    """Create one tool-use preference pair.

    The chosen side teaches the complete tool path. The rejected side intentionally
    answers directly without calling the PubMed tool. That contrast targets the
    exact failure mode observed in OpenWebUI: the model can answer, but skips the
    available PubMed research tool unless explicitly told to use it.
    """
    chosen = make_tool_sft_row(example)["messages"]
    system_prompt = chosen[0]["content"]
    rejected = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.question},
        {"role": "assistant", "content": example.answer},
    ]
    return {
        "chosen": chosen,
        "rejected": rejected,
        "tools": [TOOL_SPEC],
        "source": "tool_calling_augmentation_direct_answer_reject",
        "source_file": example.source_file,
        "source_index": example.source_index,
        "cancer_type": example.cancer_type,
    }


def collect_examples(source_dir: Path, max_examples: int, seed: int) -> list[ToolExample]:
    """Collect bounded examples from validated QA first, falling back to top-level SFT."""
    candidate_files = sorted((source_dir / DATASET_NAME / "qa_validated").glob("*.jsonl"))
    if not candidate_files:
        candidate_files = [source_dir / ORIGINAL_SFT_FILENAME]

    rng = random.Random(seed)
    rng.shuffle(candidate_files)

    examples: list[ToolExample] = []
    for path in candidate_files:
        if not path.exists():
            continue
        for row_index, row in enumerate(iter_jsonl(path), start=1):
            parts = extract_conversation_parts(row)
            if not parts:
                continue
            system_prompt, question, answer = parts
            cancer_type = row.get("cancer_type") or path.stem
            query = build_pubmed_query(question, cancer_type)
            call_id = stable_id(path.name, str(row_index), question, prefix="call_pubmed")
            tool_result = build_tool_result(len(examples) + 1, query, answer, cancer_type)
            examples.append(
                ToolExample(
                    source_index=row_index,
                    source_file=str(path.relative_to(source_dir)),
                    cancer_type=cancer_type,
                    question=question,
                    answer=answer,
                    system_prompt=system_prompt,
                    query=query,
                    tool_result=tool_result,
                    call_id=call_id,
                )
            )
            if max_examples > 0 and len(examples) >= max_examples:
                rng.shuffle(examples)
                return examples

    rng.shuffle(examples)
    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


def copy_backup_tree(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Restore the original training-data tree from the read-only backup."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to recreate it from the backup."
            )
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir, ignore=shutil.ignore_patterns(".cache"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore PubMed training data and add deterministic tool-calling augmentation files."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-tool-examples",
        type=int,
        default=None,
        help="Explicit tool example cap. By default this is derived from DPO_MAX_PAIRS in the DPO notebook.",
    )
    parser.add_argument(
        "--tool-ratio-to-chosen",
        type=float,
        default=1.0,
        help="Tool examples per selected DPO pair when --max-tool-examples is not set. Default: 1.0.",
    )
    parser.add_argument("--sft-notebook", type=Path, default=DEFAULT_SFT_NOTEBOOK)
    parser.add_argument("--dpo-notebook", type=Path, default=DEFAULT_DPO_NOTEBOOK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Backup source directory does not exist: {source_dir}")
    if not (source_dir / ORIGINAL_SFT_FILENAME).exists():
        raise FileNotFoundError(f"Missing original SFT file: {source_dir / ORIGINAL_SFT_FILENAME}")
    if not (source_dir / ORIGINAL_DPO_FILENAME).exists():
        raise FileNotFoundError(f"Missing original DPO file: {source_dir / ORIGINAL_DPO_FILENAME}")

    tool_example_limit, limit_details = resolve_tool_example_limit(
        source_dir=source_dir,
        max_tool_examples=args.max_tool_examples,
        tool_ratio_to_chosen=args.tool_ratio_to_chosen,
        sft_notebook=args.sft_notebook.resolve(),
        dpo_notebook=args.dpo_notebook.resolve(),
    )

    print(f"Restoring original training data from: {source_dir}")
    print(f"Writing active training data to:       {output_dir}")
    print(f"Tool example limit:                   {tool_example_limit:,} ({limit_details['limit_reason']})")
    print(f"  SFT selected rows:                  {limit_details['sft_selected_rows']:,}")
    print(f"  DPO selected pairs:                 {limit_details['dpo_selected_pairs']:,}")
    copy_backup_tree(source_dir, output_dir, overwrite=args.overwrite)

    print("Collecting deterministic tool-calling examples...")
    examples = collect_examples(source_dir, max_examples=tool_example_limit, seed=args.seed)
    if not examples:
        raise RuntimeError("No source examples found for tool-calling augmentation.")

    tool_sft_path = output_dir / TOOL_SFT_FILENAME
    tool_dpo_path = output_dir / TOOL_DPO_FILENAME
    sft_count = write_jsonl(tool_sft_path, (make_tool_sft_row(example) for example in examples))
    dpo_count = write_jsonl(tool_dpo_path, (make_tool_dpo_row(example) for example in examples))

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "original_sft_file": str(output_dir / ORIGINAL_SFT_FILENAME),
        "original_dpo_file": str(output_dir / ORIGINAL_DPO_FILENAME),
        "tool_sft_file": str(tool_sft_path),
        "tool_dpo_file": str(tool_dpo_path),
        "tool_name": TOOL_NAME,
        "tool_examples": len(examples),
        "limit_details": limit_details,
        "sft_rows_written": sft_count,
        "dpo_rows_written": dpo_count,
        "seed": args.seed,
        "notes": [
            "No network calls were made.",
            "Original backup data was copied, not modified.",
            "Synthetic tool results use TRAINING-SNAPSHOT identifiers instead of fabricated PMIDs.",
            "Use tool_sft_file for the supplemental tool-calling SFT notebook.",
            "Use tool_dpo_file, or a merge of original_dpo_file plus tool_dpo_file, for tool-aware DPO.",
        ],
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  Original data restored: {output_dir}")
    print(f"  Tool SFT rows:          {sft_count:,} -> {tool_sft_path}")
    print(f"  Tool DPO rows:          {dpo_count:,} -> {tool_dpo_path}")
    print(f"  Manifest:               {output_dir / MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())