#!/usr/bin/env python3
"""Evaluate base and LoRA vLLM models on the pinned VQA-RAD test split."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import string
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


PUBMED_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = (
    PUBMED_ROOT
    / "data/evaluation/oncology-medgemma27b-multimodal-v1/vqa-rad/data"
    / "test-00000-of-00001-e5bc3d208bb4deeb.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PUBMED_ROOT
    / "data/evaluation/oncology-medgemma27b-multimodal-v1/vqa-rad/results"
)
DEFAULT_MODELS = (
    "medgemma:27b-it-q4_K_M",
    "Oncologist_multimodel_canary",
)
DATASET_REVISION = "bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9"
DATASET_SHA256 = "eb520bdab1116dd4f420120da19049d2315389fa126d031f65ec42e153264ea7"
SYSTEM_PROMPT = (
    "Answer the user's question using only the supplied medical image. "
    "Return only the shortest answer that directly answers the question. "
    "If the image does not support an answer, reply unknown."
)
PUNCTUATION_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(PUNCTUATION_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def image_data_url(image_value: dict[str, Any]) -> tuple[str, str]:
    image_bytes = image_value.get("bytes")
    if not image_bytes:
        raise ValueError("VQA-RAD row does not contain embedded image bytes")
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = (image.format or "JPEG").lower()
    media_type = "image/jpeg" if image_format in {"jpg", "jpeg"} else f"image/{image_format}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    return f"data:{media_type};base64,{encoded}", image_sha256


def request_completion(
    endpoint: str,
    model: str,
    question: str,
    data_url: str,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any] | None, str | None, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        content = str(result["choices"][0]["message"]["content"] or "").strip()
        return content, result.get("usage"), None, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return "", None, f"HTTP {exc.code}: {body}", time.monotonic() - started
    except Exception as exc:
        return "", None, repr(exc), time.monotonic() - started


def evaluate_one(
    endpoint: str,
    model: str,
    index: int,
    row: dict[str, Any],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    data_url, image_sha256 = image_data_url(row["image"])
    question = str(row["question"]).strip()
    reference = str(row["answer"]).strip()
    response, usage, error, elapsed_seconds = request_completion(
        endpoint, model, question, data_url, max_tokens, timeout
    )
    normalized_reference = normalize_answer(reference)
    normalized_response = normalize_answer(response)
    answer_type = "closed" if normalized_reference in {"yes", "no"} else "open"
    return {
        "key": f"{model}:{index}",
        "index": index,
        "model": model,
        "image_sha256": image_sha256,
        "question": question,
        "reference": reference,
        "response": response,
        "normalized_reference": normalized_reference,
        "normalized_response": normalized_response,
        "answer_type": answer_type,
        "exact_match": error is None and normalized_response == normalized_reference,
        "error": error,
        "elapsed_seconds": elapsed_seconds,
        "usage": usage,
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result = json.loads(line)
                existing[result["key"]] = result
    return existing


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({result["model"] for result in results}):
        model_results = [result for result in results if result["model"] == model]
        model_summary: dict[str, Any] = {
            "total": len(model_results),
            "errors": sum(result["error"] is not None for result in model_results),
        }
        for answer_type in ("closed", "open", "overall"):
            selected = (
                model_results
                if answer_type == "overall"
                else [result for result in model_results if result["answer_type"] == answer_type]
            )
            correct = sum(result["exact_match"] for result in selected)
            model_summary[answer_type] = {
                "correct": correct,
                "total": len(selected),
                "accuracy": correct / len(selected) if selected else None,
            }
        summary[model] = model_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if sha256_file(dataset) != DATASET_SHA256:
        raise ValueError(f"Dataset hash mismatch: {dataset}")
    rows = pq.read_table(dataset).to_pylist()
    if len(rows) != 451:
        raise ValueError(f"Expected 451 rows, found {len(rows)}")
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "responses.jsonl"
    summary_path = output_dir / "summary.json"
    existing = {} if args.overwrite else load_existing(results_path)

    tasks = [
        (model, index, row)
        for model in args.models
        for index, row in enumerate(rows)
        if f"{model}:{index}" not in existing
    ]
    print(f"Rows: {len(rows)} | Models: {len(args.models)} | Pending requests: {len(tasks)}")

    if not tasks and summary_path.exists():
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps(report["summary"], indent=2))
        print(f"Responses: {results_path}")
        print(f"Summary: {summary_path}")
        return 1 if any(result["error"] for result in existing.values()) else 0

    with results_path.open("a" if existing else "w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_one,
                    args.endpoint,
                    model,
                    index,
                    row,
                    args.max_tokens,
                    args.timeout,
                ): (model, index)
                for model, index, row in tasks
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                existing[result["key"]] = result
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                completed += 1
                state = "ERROR" if result["error"] else ("PASS" if result["exact_match"] else "MISS")
                print(f"[{completed:03d}/{len(tasks):03d}] {state} {result['model']} row={result['index']}")

    selected_keys = {
        f"{model}:{index}" for model in args.models for index in range(len(rows))
    }
    results = [existing[key] for key in selected_keys if key in existing]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_repo": "flaviagiammarino/vqa-rad",
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "system_prompt": SYSTEM_PROMPT,
        "generation": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "endpoint": args.endpoint,
        },
        "normalization": "NFKC + casefold + ASCII punctuation-to-space + whitespace collapse",
        "models": args.models,
        "requested_rows": len(rows),
        "summary": summarize(results),
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Responses: {results_path}")
    print(f"Summary: {summary_path}")
    return 1 if any(result["error"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())