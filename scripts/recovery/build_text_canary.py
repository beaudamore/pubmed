#!/usr/bin/env python3
"""Build a source-grounded prompt/completion canary for MedGemma SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source-clean"
DEFAULT_QA_DIR = (
    PROJECT_ROOT
    / "data"
    / "training-data-backups-before-tool-calling-augmentation"
    / "pubmed_oncologist_v2"
    / "qa_validated"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training-data" / "recovery" / "text_canary_512"

SYSTEM_PROMPT = """You are an oncology-focused medical language model for educational and research support.

Evidence is conditional. Treat an abstract, tool result, or image analysis as available only when it is explicitly present in the conversation. Never invent a PMID, trial name, statistic, guideline version, measurement, retrieval result, or image finding. When evidence is missing, ask for it or state the limitation. Answer greetings and unrelated harmless questions normally without assuming a medical context. Do not provide a diagnosis or replace professional medical care."""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "this", "to", "what",
    "when", "which", "with",
}

GENERAL_REPLAY = (
    ("Hi", "Hi! How can I help you today?"),
    ("Hello there", "Hello! What would you like help with?"),
    ("Thanks for your help.", "You're welcome."),
    ("Can you explain something simply?", "Yes. Tell me the topic, and I'll explain it in clear, straightforward terms."),
    ("I am not sure how to phrase my question.", "That's fine. Share what you know so far, and I can help you clarify the question."),
    ("Summarize this for me.", "Please provide the text you want summarized."),
    ("What information do you need from me?", "Please share the question, text, or context you want me to work with."),
    ("Can you compare two documents?", "Yes. Provide both documents and tell me which aspects you want compared."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Write a short checklist for reviewing a document.", "Check the purpose, evidence, assumptions, internal consistency, missing information, and final conclusions."),
    ("Can you help brainstorm research questions?", "Yes. Share the topic, audience, and constraints, and I can propose focused research questions."),
    ("I only have part of the source material.", "Share the portion you have. I can analyze it while clearly identifying what cannot be determined from the missing material."),
    ("Please answer only from information I provide.", "Understood. I will use only the supplied information and will identify anything that cannot be established from it."),
    ("Can you make up missing details?", "No. I can identify the missing details or suggest how to obtain them, but I should not present invented information as fact."),
    ("What should I do if a source contradicts itself?", "Identify the conflicting claims, compare their supporting evidence, and avoid choosing between them without a justified basis."),
    ("Goodbye", "Goodbye."),
)

MISSING_EVIDENCE_REQUESTS = (
    "Summarize the PubMed abstract I provided.",
    "What were the primary outcomes in the article above?",
    "Extract the trial design and response rates from that study.",
    "Which biomarkers were significant in the abstract I shared?",
    "Compare the treatment arms in the supplied paper.",
    "What limitations did the authors report?",
    "Give me the PMID and conclusions from the article.",
    "Does the provided abstract support changing clinical practice?",
    "Explain the image findings I uploaded.",
    "What did the pathology image show?",
    "Use the search results to answer my question.",
    "What did the literature tool return?",
    "List the statistics reported in the study above.",
    "Which guideline version was included in the evidence?",
    "Tell me whether the trial met its primary endpoint.",
    "What conclusion should I draw from the missing source?",
)

MISSING_EVIDENCE_RESPONSES = (
    "I don't see an abstract in this conversation. Please paste the abstract you want summarized.",
    "No article text is present here. Please provide the article or abstract before I identify its outcomes.",
    "I don't have the study text or a tool result, so I cannot extract its design or response rates. Please supply the evidence.",
    "No abstract was supplied. Please provide it so I can identify which biomarkers, if any, were reported as significant.",
    "I don't see a paper or treatment-arm data in the conversation. Please provide the source before I compare the arms.",
    "The source text is missing, so I cannot determine which limitations the authors reported. Please share the relevant text.",
    "I cannot identify a PMID or conclusion without the article information. Please provide the citation or abstract.",
    "No abstract is available in the conversation. Please share it before asking whether its findings support a practice change.",
    "I don't see a successful vision-observation report or image analysis in the conversation, so I cannot describe image findings.",
    "No pathology image observations are available here. Please provide the image through the configured vision workflow.",
    "There is no tool-result message in the conversation. I cannot claim that a search occurred or use results that were not supplied.",
    "No literature-tool response is present. Please run the tool or provide its returned evidence.",
    "The study text is absent, so I cannot report its statistics. Please provide the source.",
    "No guideline evidence or version is present in the conversation. I cannot infer one.",
    "I don't have the trial report, so I cannot determine whether its primary endpoint was met. Please supply the evidence.",
    "A conclusion cannot be grounded in a source that is not present. Please provide the source or ask a general question instead.",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_hidden_reasoning(text: str) -> str:
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def select_extractive_answer(passage: str, question: str) -> str | None:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", passage) if part.strip()]
    question_terms = content_terms(question)
    ranked = []
    for index, sentence in enumerate(sentences):
        overlap = len(question_terms & content_terms(sentence))
        if overlap:
            ranked.append((overlap, -index, sentence))
    if not ranked:
        return None
    return max(ranked)[2]


def load_sources(source_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    by_cancer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(source_dir.glob("pubmed_*.jsonl"))
    for path in paths:
        for row in iter_jsonl(path):
            passage = row.get("passage")
            if isinstance(passage, str) and passage.strip():
                by_cancer[path.stem].append(row)
    return by_cancer, paths


def unique_source_match(chunk_key: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not chunk_key:
        return None
    matches = [row for row in candidates if str(row.get("passage") or "").startswith(chunk_key)]
    return matches[0] if len(matches) == 1 else None


def grounded_candidates(source_dir: Path, qa_dir: Path) -> tuple[list[dict[str, Any]], list[Path], Counter[str]]:
    sources, source_paths = load_sources(source_dir)
    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    qa_paths = sorted(qa_dir.glob("pubmed_*.jsonl"))

    for qa_path in qa_paths:
        source_rows = sources.get(qa_path.stem, [])
        for row in iter_jsonl(qa_path):
            if row.get("grounding_verdict") != "grounded":
                rejection_counts["not_grounded"] += 1
                continue
            source = unique_source_match(str(row.get("chunk_key") or ""), source_rows)
            if source is None:
                rejection_counts["source_join_not_unique"] += 1
                continue
            source_id = str(source.get("id") or "")
            if not source_id or source_id in seen_source_ids:
                rejection_counts["duplicate_or_missing_source_id"] += 1
                continue
            question = row.get("question")
            passage = str(source.get("passage") or "").strip()
            if not isinstance(question, str) or not question.strip() or not passage:
                rejection_counts["missing_required_text"] += 1
                continue
            answer = select_extractive_answer(passage, question)
            if answer is None:
                rejection_counts["no_lexical_evidence_match"] += 1
                continue
            seen_source_ids.add(source_id)
            cancer_type = str(row.get("cancer_type") or qa_path.stem)
            candidates.append(
                {
                    "source_id": source_id,
                    "source_file": str((source_dir / f"{qa_path.stem}.jsonl").relative_to(PROJECT_ROOT)),
                    "cancer_type": cancer_type,
                    "question": question.strip(),
                    "answer": answer,
                    "passage": passage,
                    "title": str(source.get("title") or "").strip(),
                }
            )
    return candidates, source_paths + qa_paths, rejection_counts


def make_grounded_record(candidate: dict[str, Any]) -> dict[str, Any]:
    user_content = (
        '<evidence source="pubmed" status="supplied">\n'
        f"{candidate['passage']}\n"
        "</evidence>\n\n"
        "<user_request>\n"
        "Using only the supplied evidence, quote the single sentence most relevant to this question. "
        "Do not add facts or interpretation:\n"
        f"{candidate['question']}\n"
        "</user_request>"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "completion": [{"role": "assistant", "content": candidate["answer"]}],
        "record_type": "grounded_abstract",
        "source_id": candidate["source_id"],
        "source_file": candidate["source_file"],
        "cancer_type": candidate["cancer_type"],
        "evidence_supplied": True,
        "tool_expected": False,
    }


def make_replay_records() -> list[dict[str, Any]]:
    records = []
    for index, (user, assistant) in enumerate(GENERAL_REPLAY):
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "completion": [{"role": "assistant", "content": assistant}],
                "record_type": "general_replay",
                "source_id": f"general_replay_{index:02d}",
                "source_file": "authored_recovery_fixture",
                "cancer_type": None,
                "evidence_supplied": False,
                "tool_expected": False,
            }
        )
    for index, (user, assistant) in enumerate(zip(MISSING_EVIDENCE_REQUESTS, MISSING_EVIDENCE_RESPONSES, strict=True)):
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "completion": [{"role": "assistant", "content": assistant}],
                "record_type": "missing_evidence",
                "source_id": f"missing_evidence_{index:02d}",
                "source_file": "authored_recovery_fixture",
                "cancer_type": None,
                "evidence_supplied": False,
                "tool_expected": False,
            }
        )
    return records


def balanced_sample(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["cancer_type"]].append(candidate)
    rng = random.Random(seed)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for cancer_type, rows in grouped.items():
        rng.shuffle(rows)
        queues[cancer_type] = deque(rows)

    selected: list[dict[str, Any]] = []
    cancer_types = sorted(queues)
    while len(selected) < count:
        progressed = False
        for cancer_type in cancer_types:
            if queues[cancer_type] and len(selected) < count:
                selected.append(queues[cancer_type].popleft())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} grounded records but only selected {len(selected)}")
    return selected


def validate_record(record: dict[str, Any]) -> None:
    if set(record) < {"prompt", "completion", "record_type", "source_id"}:
        raise ValueError(f"Record is missing required fields: {record.get('source_id')}")
    prompt = record["prompt"]
    completion = record["completion"]
    if not isinstance(prompt, list) or [message.get("role") for message in prompt] != ["system", "user"]:
        raise ValueError(f"Invalid prompt roles: {record.get('source_id')}")
    if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError(f"Invalid completion: {record.get('source_id')}")
    if not all(isinstance(message.get("content"), str) and message["content"].strip() for message in prompt + completion):
        raise ValueError(f"Empty message content: {record.get('source_id')}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-grounded prompt/completion text canary.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    qa_dir = args.qa_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.is_dir() or not qa_dir.is_dir():
        raise FileNotFoundError("Both --source-dir and --qa-dir must exist")
    replay_records = make_replay_records()
    if args.size <= len(replay_records):
        raise ValueError(f"--size must exceed the {len(replay_records)} fixed replay records")

    candidates, input_paths, rejection_counts = grounded_candidates(source_dir, qa_dir)
    grounded_count = args.size - len(replay_records)
    selected = balanced_sample(candidates, grounded_count, args.seed)
    records = [make_grounded_record(candidate) for candidate in selected] + replay_records
    random.Random(args.seed).shuffle(records)
    for record in records:
        validate_record(record)

    output_path = output_dir / "train.jsonl"
    write_jsonl(output_path, records)
    record_counts = Counter(record["record_type"] for record in records)
    cancer_counts = Counter(record["cancer_type"] for record in records if record["cancer_type"])
    manifest = {
        "status": "complete",
        "dataset_type": "prompt_completion_text_canary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(records),
        "seed": args.seed,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "record_counts": dict(record_counts),
        "cancer_counts": dict(sorted(cancer_counts.items())),
        "candidate_rows": len(candidates),
        "rejection_counts": dict(rejection_counts),
        "input_files": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
        "tool_records_included": False,
        "tool_protocol_gate": "blocked: production Hermes parser did not parse Gemma tool-call text and the saved chat template discarded structured tool_calls",
        "notes": [
            "Every grounded row uses one uniquely joined source passage with grounding_verdict=grounded.",
            "Every grounded completion is an exact sentence selected from the supplied evidence by deterministic lexical overlap.",
            "The canary tests evidence boundaries and behavior; it is not a clinical-reasoning or production-quality dataset.",
            "No unrelated conversations are concatenated or packed.",
            "This canary must not be deployed as a production model.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Candidates: {len(candidates):,}")
    print(f"Rows written: {len(records):,}")
    print(f"Record counts: {dict(record_counts)}")
    print(f"Output: {output_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())