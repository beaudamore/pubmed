# PubMed MedGemma V3 SFT Tune Success Report

Date: 2026-07-13
Notebook: training/pubmed/notebooks/loras/medgemma/v3/pubmed_sft_training_medgemma_v3.ipynb

## Executive Summary
The SFT run completed successfully from setup through training, adapter export, inference smoke tests, and cold-reload validation.

## What Ran Successfully
- Environment setup completed with GPU available (NVIDIA GB10) and required libraries loaded.
- Base model loaded successfully: unsloth/medgemma-27b-text-it-unsloth-bnb-4bit.
- Dataset loading and validation completed with no structural failures.
- Dataset formatting and manual packing completed with near-zero token waste.
- LoRA adapters were attached and training completed for 1 epoch.
- LoRA artifacts were saved and present on disk.
- Inference smoke tests produced coherent oncology responses across multiple cancer prompt contexts.
- Cold reload test confirmed adapters load cleanly from disk and are reusable for Phase 2 DPO.

## Key Output Metrics

### Setup and Configuration
- Torch: 2.10.0a0+b558c986e8.nv25.11
- CUDA toolkit reported: 13.0
- Unsloth: 2026.5.7
- Transformers: 5.10.0.dev0
- TRL: 0.24.0
- Flex attention override env var set: UNSLOTH_ENABLE_FLEX_ATTENTION=0

### Data and Packing
- Raw file rows detected: 19,293
- SFT cap applied: 9,000 conversations
- Cancer types represented: 11
- Unique system prompts extracted: 11
- Bad structure rows: 0
- Empty final responses: 0
- Rows with explicit tool calls: 4,552 (50.6%)
- Rows without explicit tool calls: 4,448 (49.4%)
- Packed dataset result: 1,715 chunks at sequence length 16,384
- Total packed tokens: 28,105,181
- Tail waste: 6,621 tokens (~0.0%)

### Model and LoRA
- GPU memory after model load: ~16.6 GB allocated
- LoRA config: r=32, alpha=32, dropout=0
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params reported: 227,033,088

### Training Outcome
- Start mode: fresh run (no checkpoint resume)
- Planned/actual steps: 215
- Epochs: 1
- Final reported training_loss: 1.0030
- Reported train_runtime: 1502.4 minutes

## Saved Artifact Verification
Verified in output directory:
- adapter_model.safetensors: ~867 MB
- tokenizer.json: ~32 MB
- tokenizer.model: ~4.5 MB
- tokenizer_config.json: ~1.2 MB
- adapter_config.json, chat_template.jinja, README.md
- oncologist_system_prompts.json

Output location:
/home/spark/projects/training/pubmed/output/v3/pubmed_oncologist_v3_medgemma_sft/lora_adapters

## Inference and Reload Validation
- Inference smoke test ran across 3 cancer prompt contexts and generated full model responses.
- Cold-reload test loaded adapter from disk and generated a valid response.
- Notebook explicitly reports that the adapter path is ready for the Phase 2 DPO notebook.

## Non-blocking Warnings Observed
- Repeated generation warning: max_new_tokens takes precedence over max_length.
- Deprecation warnings from AutoAWQ and multiprocessing fork warning were present.
- No warning observed prevented training completion or adapter export.

## Conclusion
This MedGemma 27B 4-bit QLoRA SFT run is a successful tune. Training completed, artifacts were saved, and adapter portability was validated.
