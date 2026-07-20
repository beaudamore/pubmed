# Stage 01: Text Canary

Status: complete

The deterministic 512-record canary established that clean targets and native
completion-only loss recover ordinary conversation, missing-evidence behavior,
and exact evidence extraction.

Canonical artifacts:

- Dataset: `data/training-data/recovery/text_canary_512/`
- Builder: `scripts/recovery/build_text_canary.py`
- Notebook: `notebooks/loras/medgemma/recovery/pubmed_medgemma_text_canary_sft.ipynb`
- Evaluator: `scripts/recovery/evaluate_text_canary.py`
- Output: `output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_sft/`

Exit evidence:

- 64/64 optimizer steps.
- Cold reload succeeded.
- Standalone behavioral gate: 18/18.
- Direct-vLLM behavioral gate: 18/18.
