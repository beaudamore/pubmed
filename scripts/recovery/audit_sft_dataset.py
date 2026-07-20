#!/usr/bin/env python3
"""Audit oncology SFT assistant targets before conversion or training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "training-data" / "pubmed_oncologist_v2_tool_sft_messages.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "training-data" / "recovery" / "sft_audit_report.json"

ABSTRACT_CLAIM_PATTERNS = (
    re.compile(r"\b(?:the|this) (?:provided|shared|supplied|above) (?:pubmed )?abstract\b", re.IGNORECASE),
    re.compile(r"\b(?:abstract|article|study) (?:you|the user) (?:provided|shared|supplied)\b", re.IGNORECASE),
    re.compile(r"\bbased on (?:the|this|your) (?:provided |shared |supplied )?(?:pubmed )?abstract\b", re.IGNORECASE),
)
IMAGE_CLAIM_PATTERNS = (
    re.compile(r"\b(?:the|this|your) (?:provided|shared|uploaded) image\b", re.IGNORECASE),
    re.compile(r"\bimage (?:you|the user) (?:provided|shared|uploaded)\b", re.IGNORECASE),
)
TOOL_CLAIM_PATTERNS = (
    re.compile(r"\b(?:the|this) (?:tool|search|retrieval|database) (?:result|response|output)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:retrieved|returned) (?:evidence|results?|articles?)\b", re.IGNORECASE),
)
ABSTRACT_EVIDENCE_PATTERNS = (
    re.compile(r"\babstract\s*:", re.IGNORECASE),
    re.compile(r"\bpubmed\b.{0,80}\b(?:abstract|pmid)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"continue this clinical/research text", re.IGNORECASE),
)
IMAGE_EVIDENCE_PATTERN = re.compile(r"<vision_observations\b", re.IGNORECASE)
SYNTHETIC_IDENTIFIER_PATTERN = re.compile(r"\bTRAINING-SNAPSHOT-[0-9]+\b", re.IGNORECASE)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
            yield line_number, row


def normalized_opening(text: str, words: int = 8) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"[#*_>`~-]+", " ", text)
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(tokens[:words])


def contains_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def has_abstract_evidence(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") not in {"user", "tool"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and contains_any(content, ABSTRACT_EVIDENCE_PATTERNS):
            return True
    return False


def has_image_evidence(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(message.get("content"), str)
        and IMAGE_EVIDENCE_PATTERN.search(message["content"])
        for message in messages
    )


def audit_row(line_number: int, row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ([{"line": line_number, "code": "invalid_messages", "assistant_index": None}], [])

    findings: list[dict[str, Any]] = []
    openings: list[str] = []
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    valid_roles = {"system", "user", "assistant", "tool"}
    if len(roles) != len(messages) or any(role not in valid_roles for role in roles):
        findings.append({"line": line_number, "code": "invalid_roles", "assistant_index": None, "roles": roles})

    for assistant_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        prior = messages[:assistant_index]
        content = message.get("content") or ""
        if not isinstance(content, str):
            findings.append({"line": line_number, "code": "non_string_assistant_content", "assistant_index": assistant_index})
            continue
        if content.strip():
            opening = normalized_opening(content)
            if opening:
                openings.append(opening)
        if contains_any(content, ABSTRACT_CLAIM_PATTERNS) and not has_abstract_evidence(prior):
            findings.append({"line": line_number, "code": "phantom_abstract", "assistant_index": assistant_index})
        if contains_any(content, IMAGE_CLAIM_PATTERNS) and not has_image_evidence(prior):
            findings.append({"line": line_number, "code": "phantom_image", "assistant_index": assistant_index})
        if contains_any(content, TOOL_CLAIM_PATTERNS) and not any(item.get("role") == "tool" for item in prior if isinstance(item, dict)):
            findings.append({"line": line_number, "code": "phantom_tool_result", "assistant_index": assistant_index})
        if SYNTHETIC_IDENTIFIER_PATTERN.search(content):
            findings.append({"line": line_number, "code": "synthetic_identifier_in_target", "assistant_index": assistant_index})

    return findings, openings


def audit(path: Path, top_openings: int) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    openings: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    row_count = 0
    assistant_actions = 0

    for line_number, row in iter_jsonl(path):
        row_count += 1
        messages = row.get("messages")
        if isinstance(messages, list):
            role_sequences[" -> ".join(str(message.get("role")) for message in messages if isinstance(message, dict))] += 1
            assistant_actions += sum(message.get("role") == "assistant" for message in messages if isinstance(message, dict))
        source_counts[str(row.get("source") or "unknown")] += 1
        row_findings, row_openings = audit_row(line_number, row)
        findings.extend(row_findings)
        openings.update(row_openings)

    finding_counts = Counter(item["code"] for item in findings)
    blocking_codes = {
        "invalid_messages",
        "invalid_roles",
        "non_string_assistant_content",
        "phantom_abstract",
        "phantom_image",
        "phantom_tool_result",
        "synthetic_identifier_in_target",
    }
    blocking_count = sum(count for code, count in finding_counts.items() if code in blocking_codes)
    return {
        "input_file": str(path),
        "rows": row_count,
        "assistant_actions": assistant_actions,
        "passed": blocking_count == 0,
        "blocking_findings": blocking_count,
        "finding_counts": dict(sorted(finding_counts.items())),
        "source_counts": dict(source_counts.most_common()),
        "role_sequences": dict(role_sequences.most_common()),
        "top_openings": [{"opening": opening, "count": count} for opening, count in openings.most_common(top_openings)],
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit oncology SFT targets against their preceding context.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-openings", type=int, default=25)
    parser.add_argument("--allow-findings", action="store_true", help="Write the report but return success despite blocking findings.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    report_path = args.report.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if args.top_openings <= 0:
        raise ValueError("--top-openings must be positive")

    report = audit(input_path, args.top_openings)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Rows: {report['rows']:,}")
    print(f"Assistant actions: {report['assistant_actions']:,}")
    print(f"Blocking findings: {report['blocking_findings']:,}")
    for code, count in report["finding_counts"].items():
        print(f"  {code}: {count:,}")
    print(f"Report: {report_path}")
    if report["passed"] or args.allow_findings:
        return 0
    print("FAILED: blocking target contamination exists.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())