#!/usr/bin/env python3
"""Evaluate the MedGemma text-grounding canary after a cold adapter reload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import unsloth
import torch
from unsloth import FastLanguageModel


PROJECT_ROOT = Path("/workspace/training/pubmed")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_sft/lora_adapters"
)
DEFAULT_DATASET = PROJECT_ROOT / "data/training-data/recovery/text_canary_512/train.jsonl"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data/source-clean"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_sft/evaluation/report.json"
)

SYSTEM_PROMPT = """You are an oncology-focused medical language model for educational and research support.

Evidence is conditional. Treat an abstract, tool result, or image analysis as available only when it is explicitly present in the conversation. Never invent a PMID, trial name, statistic, guideline version, measurement, retrieval result, or image finding. When evidence is missing, ask for it or state the limitation. Answer greetings and unrelated harmless questions normally without assuming a medical context. Do not provide a diagnosis or replace professional medical care."""

PHANTOM_PATTERNS = (
    re.compile(r"\b(?:the|this) (?:provided|shared|supplied) (?:pubmed )?abstract\b", re.IGNORECASE),
    re.compile(r"\b(?:the|this) (?:tool|search|retrieval) (?:result|response|output)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:the|this|your) (?:provided|shared|uploaded) image\b", re.IGNORECASE),
)
MISSING_CUES = re.compile(
    r"\b(?:no|not|don't|do not|cannot|can't|missing|provide|supply|share|paste|absent|unavailable)\b",
    re.IGNORECASE,
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def first_abstract_sentence(row: dict[str, Any]) -> str | None:
    abstract = str(row.get("abstract") or "").strip()
    if not abstract:
        return None
    sentence = re.split(r"(?<=[.!?])\s+|\n+", abstract, maxsplit=1)[0].strip()
    return sentence if len(sentence.split()) >= 6 else None


def sentence_prefix(sentence: str, word_count: int = 8) -> str:
    words = sentence.split()
    return " ".join(words[: min(word_count, len(words))])


def build_behavior_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "greeting",
            "category": "general",
            "user": "Hi",
            "check": "general_no_phantom",
        },
        {
            "id": "general_capabilities",
            "category": "general",
            "user": "What can you help me with?",
            "check": "general_no_phantom",
        },
        {
            "id": "out_of_domain_fact",
            "category": "general",
            "user": "What is the capital of France?",
            "check": "contains",
            "expected": "Paris",
        },
        {
            "id": "missing_abstract",
            "category": "missing_evidence",
            "user": "Summarize the PubMed abstract I provided.",
            "check": "missing_evidence",
        },
        {
            "id": "missing_statistics",
            "category": "missing_evidence",
            "user": "List the response rates reported in the study above.",
            "check": "missing_evidence",
        },
        {
            "id": "missing_tool_result",
            "category": "missing_evidence",
            "user": "What did the literature search tool return?",
            "check": "missing_evidence",
        },
        {
            "id": "missing_image",
            "category": "missing_evidence",
            "user": "What abnormalities are visible in the pathology image I uploaded?",
            "check": "missing_evidence",
        },
        {
            "id": "underspecified_summary",
            "category": "missing_evidence",
            "user": "Summarize this for me.",
            "check": "missing_evidence",
        },
    ]


def build_heldout_cases(source_dir: Path, dataset_path: Path) -> list[dict[str, Any]]:
    training_ids = {
        str(row.get("source_id"))
        for row in iter_jsonl(dataset_path)
        if row.get("record_type") == "grounded_abstract"
    }
    cases = []
    for source_path in sorted(source_dir.glob("pubmed_*.jsonl")):
        selected = None
        for row in iter_jsonl(source_path):
            source_id = str(row.get("id") or "")
            sentence = first_abstract_sentence(row)
            passage = str(row.get("passage") or "").strip()
            if source_id and source_id not in training_ids and sentence and sentence in passage:
                selected = (source_id, sentence, passage)
                break
        if selected is None:
            raise RuntimeError(f"No held-out source record found in {source_path}")
        source_id, expected, passage = selected
        prefix = sentence_prefix(expected)
        cases.append(
            {
                "id": f"heldout_{source_path.stem}",
                "category": "heldout_evidence",
                "source_id": source_id,
                "user": (
                    '<evidence source="pubmed" status="supplied">\n'
                    f"{passage}\n"
                    "</evidence>\n\n"
                    "<user_request>\n"
                    "Quote exactly the complete sentence that starts with these words:\n"
                    f"{prefix}\n"
                    "Do not add facts or interpretation.\n"
                    "</user_request>"
                ),
                "check": "evidence_quote",
                "expected": expected,
                "evidence": passage,
            }
        )
    return cases


def generate(model: Any, tokenizer: Any, user: str, max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[0, encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_case(case: dict[str, Any], response: str) -> tuple[bool, list[str]]:
    failures = []
    normalized_response = normalize_text(response)
    check = case["check"]
    if not response:
        failures.append("empty_response")
    if check == "general_no_phantom":
        if any(pattern.search(response) for pattern in PHANTOM_PATTERNS):
            failures.append("phantom_context")
    elif check == "contains":
        if normalize_text(str(case["expected"])) not in normalized_response:
            failures.append("missing_expected_text")
    elif check == "missing_evidence":
        if not MISSING_CUES.search(response):
            failures.append("did_not_identify_missing_evidence")
        if re.search(r"\b\d+(?:\.\d+)?%\b", response):
            failures.append("invented_percentage")
    elif check == "evidence_quote":
        expected = normalize_text(str(case["expected"]))
        if expected not in normalized_response:
            failures.append("missing_expected_evidence_sentence")
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(case["evidence"])))
        response_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", response))
        if not response_numbers <= evidence_numbers:
            failures.append("unsupported_numeric_token")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold-reload and evaluate the MedGemma text canary.")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    dataset = args.dataset.resolve()
    source_dir = args.source_dir.resolve()
    output = args.output.resolve()
    for path in (adapter, dataset, source_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    cases = build_behavior_cases() + build_heldout_cases(source_dir, dataset)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for index, case in enumerate(cases, start=1):
        response = generate(model, tokenizer, case["user"], args.max_new_tokens)
        passed, failures = evaluate_case(case, response)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "source_id": case.get("source_id"),
                "passed": passed,
                "failures": failures,
                "prompt": case["user"],
                "response": response,
                "expected": case.get("expected"),
            }
        )
        print(f"[{index:02d}/{len(cases):02d}] {'PASS' if passed else 'FAIL'} {case['id']}")

    failed = [result for result in results if not result["passed"]]
    category_counts: dict[str, dict[str, int]] = {}
    for result in results:
        counts = category_counts.setdefault(result["category"], {"passed": 0, "failed": 0})
        counts["passed" if result["passed"] else "failed"] += 1
    report = {
        "status": "passed" if not failed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "unsloth_version": unsloth.__version__,
        "total_cases": len(results),
        "passed_cases": len(results) - len(failed),
        "failed_cases": len(failed),
        "category_counts": category_counts,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "attention_mask_supplied": True,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}")
    print(f"Result: {report['status']} ({report['passed_cases']}/{report['total_cases']})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())