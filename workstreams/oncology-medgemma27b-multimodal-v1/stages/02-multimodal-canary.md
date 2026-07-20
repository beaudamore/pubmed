# Stage 02: Multimodal Canary

Status: complete

The text canary was repeated on `unsloth/medgemma-27b-it` with a language-only
LoRA. The SigLIP vision tower and multimodal projector remained frozen.

Canonical artifacts:

- Feasibility notebook: `notebooks/loras/medgemma/recovery/medgemma_27b_multimodal_4bit_feasibility.ipynb`
- Training notebook: `notebooks/loras/medgemma/recovery/pubmed_medgemma_text_canary_sft_multimodal.ipynb`
- Evaluator: `scripts/recovery/evaluate_multimodal_text_canary.py`
- Output: `output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_multimodal_sft/`
- Runtime compose: `docs/dgx-compose.yaml`

Exit evidence:

- Zero trainable vision or projector tensors.
- Cold reload succeeded.
- Standalone text gate: 18/18.
- Direct-vLLM text gate: 18/18.
- Synthetic visual checks passed before and after training.
- OpenWebUI delivered a real image to the LoRA-served multimodal model.

The OpenWebUI result proved transport, not clinical accuracy. A modality
mismatch and unsupported detailed findings triggered Stage 03.
