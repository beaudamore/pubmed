# Stage 03: Clinical Image Evaluation

Status: in progress

## Objective

Measure image-grounded behavior before adding multimodal training data. Compare
the clean multimodal base and `Oncologist_multimodel_canary` through the same
direct-vLLM request path, then confirm selected cases through OpenWebUI.

## Primary Benchmark

- Dataset: `flaviagiammarino/vqa-rad`
- Immutable revision: `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`
- Split: author-provided `test`
- Expected examples: 451
- Local data: `data/evaluation/oncology-medgemma27b-multimodal-v1/vqa-rad/`
- Test Parquet SHA-256: `eb520bdab1116dd4f420120da19049d2315389fa126d031f65ec42e153264ea7`
- Policy: evaluation-only; do not use any VQA-RAD split for this workstream's training

Verified local structure:

- 451 question-answer rows.
- 203 unique images.
- 251 closed yes/no questions.
- 200 open questions.
- Embedded image payloads decode successfully.

Evaluator: `workstreams/oncology-medgemma27b-multimodal-v1/scripts/evaluate_vqa_rad_vllm.py`

## VQA-RAD Direct-vLLM Result

Completed 2026-07-14 with identical deterministic requests to the clean base
and LoRA aliases. All 902 requests completed with no transport errors.

| Model | Closed exact | Open exact | Overall exact |
| --- | ---: | ---: | ---: |
| `medgemma:27b-it-q4_K_M` | 171/251 (68.13%) | 41/200 (20.50%) | 212/451 (47.01%) |
| `Oncologist_multimodel_canary` | 172/251 (68.53%) | 49/200 (24.50%) | 221/451 (49.00%) |

LoRA minus base:

- Closed: +1 correct, +0.40 percentage points.
- Open: +8 correct, +4.00 percentage points.
- Overall: +9 correct, +2.00 percentage points.
- Paired outcomes: 21 improved, 12 regressed, 200 remained correct, and 218 remained incorrect.
- Normalized `unknown` responses increased from 36 for base to 49 for LoRA.

The open-answer gain includes legitimate corrected answers and stricter concise
formatting. For example, references such as `mri`, `axial`, and `ivc` matched
the LoRA's concise output where longer or incorrect base outputs missed. This
strict exact metric therefore measures the frozen VQA task contract; it is not
a semantic-equivalence or clinical-correctness score.

Result artifacts:

- Responses: `data/evaluation/oncology-medgemma27b-multimodal-v1/vqa-rad/results/responses.jsonl`
- Responses SHA-256: `b1cc8077ffa32223743a6e68c369ff2c114e2cbd5069f9a6eb40a321d20f2e87`
- Summary: `data/evaluation/oncology-medgemma27b-multimodal-v1/vqa-rad/results/summary.json`
- Summary SHA-256: `51ceb9a44c7d5d4f4c40340f219a9f04602e8d670ad6bb619852a24a922309e9`

## Required Measurements

- Closed-question exact accuracy.
- Open-question normalized accuracy.
- Base-versus-LoRA result delta.
- Unsupported finding rate.
- Empty or transport-error rate.
- Per-image grouping to avoid treating repeated questions as independent images.

The exact-match, base-versus-LoRA, empty-response, and transport measurements
are complete. Unsupported-finding review and per-image aggregation remain open;
they must not be inferred from question-level exact match.

## Limitations

VQA-RAD is small and may have appeared in upstream model training. It is a
reproducible comparative gate, not proof of clinical readiness. A separate
modality-focused ROCOv2 gate and locally reviewed negative cases remain follow-up
work.

## Exit Gate

Do not define a passing threshold after seeing results. First record the frozen
test manifest, evaluator behavior, and scoring policy; then run base and LoRA
with identical deterministic settings.

Scoring uses lowercase Unicode-normalized exact match after replacing
punctuation with spaces and collapsing whitespace. Articles and clinical terms
are not removed. Raw prompts, references, responses, errors, and normalized
values are retained so alternative scoring can be applied without rerunning
inference.

