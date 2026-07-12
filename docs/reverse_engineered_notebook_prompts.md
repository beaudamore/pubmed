# Reverse Engineering Guide: Prompts to Rebuild the SFT & DPO Notebooks from Scratch

When teaching developers how to fine-tune massive models (like MedGemma 27B) on custom domain tasks while preserving advanced capabilities like selective tool usage, showing them the **final notebook** is helpful. But showing them **how to construct the notebook from scratch using generative AI prompts** is an extraordinary training technique.

This document reverse-engineers our clinical oncology SFT and DPO notebooks into two production-grade, highly-detailed prompts. Passing these prompts to models (such as Claude 3.5 Sonnet or GPT-4o) will generate clean, bug-free fine-tuning pipelines.

---

## Prompt 1: Rebuilding the SFT Notebook (`pubmed_sft_training_medgemma.ipynb`)

### Instructions to the LLM:
Copy, paste, and run the following prompt in your senior-developer generative AI workflow.

```text
Act as a Principal Deep Learning Engineer experienced in parameter-efficient fine-tuning (PEFT), Unsloth, and hardware-level optimization for extreme medical text domains. 

Your task is to generate a comprehensive, highly optimized, multi-cell Jupyter notebook matching a medical oncology fine-tuning pipeline. The notebook should fine-tune "google/medgemma-27b-text-it" (representing Phase 1 SFT) on a multi-turn conversation dataset containing tool-use sequences. 

The generated code must be robust, production-grade, self-contained, and implement the following cells and blocks in sequence:

### Cell 1: Environment Setup, Blackwell Hack & CUDA Protection
- Import: os, sys, subprocess, importlib, torch, and unsloth.
- Disable Unsloth's FlexAttention for Gemma 3 on Blackwell architecture by setting: `os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"` BEFORE any other imports. This is critical because Triton's FlexAttention backward pass overflows shared memory on sm_120.
- Check if CUDA is available. Inform the user.
- Provide a robust pip install function `_pip(*args)` to safely install 'psutil', 'matplotlib', 'ipywidgets', PIL, and 'torchvision' (with --no-deps) on standard environments without clobbering CUDA torch.
- Include a compiler builder fallback block for compiling `causal_conv1d` from source. It must run `uninstall causal-conv1d`, clear pip cache, build with `CAUSAL_CONV1D_FORCE_BUILD=TRUE`, and invalidate cache.
- Import unsloth and transformers AFTER enforcing these environment overrides.

### Cell 2: System configuration Constants
- Hardcode variables at the top of the cell:
  `BASE_LLM = "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit"` (BNB 4-bit)
  `MODEL_NAME_BASE = "pubmed_oncologist_v2_medgemma_sft"`
  `MAX_SEQ_LENGTH = 4096`
  `BATCH_SIZE = 2`, `GRAD_ACCUM = 4` (effective batch is 8)
  `LEARNING_RATE = 2e-4`, `LORA_R = 32`, `LORA_ALPHA = 32`, `LORA_DROPOUT = 0`
- Define target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
- Set test prompt: `"A 58-year-old woman with BRCA1-mutated high-grade serous ovarian cancer has progressed after platinum-based chemotherapy and a PARP inhibitor. What are the next treatment options?"`

### Cell 3: Loading the Dataset with Slicing Configurations
- Define `SFT_MAX_EXAMPLES = None` near the top of the cell (representing a local cap variable).
- Load a training JSONL from `f"{DATA_DIR}/training-data/pubmed_oncologist_v2_tool_sft_messages.jsonl"`. Support both message lists (`messages` key) and standard ShareGPT (`conversations` key) formats.
- Safely slice the processed rows if `SFT_MAX_EXAMPLES` is defined: `raw_rows = raw_rows[:SFT_MAX_EXAMPLES]`.
- Load it into a HuggingFace Dataset. Print breakdown analytics (total records, rows per cancer type and source).

### Cell 4: Validate Dataset Structures
- Verify that every conversation possesses a minimum of 3 turns, starts with a 'system' role and a 'user' role sequentionally, and ends with a non-empty 'assistant' final answer. Remove any corrupted samples.

### Cell 5: Load Model & Native Flash Attention 2 Override
- Load Model and Tokenizer via Unsloth's `FastLanguageModel.from_pretrained()`.
- Unsloth defaults Gemma 3 to native eager attention. Override eager attention with Flash Attention 2 by forcing `model.config._attn_implementation = "flash_attention_2"`. This is mandatory—eager checks matmul O(n^2) attention scaling, leading to horrible bottlenecks.
- Sanity check that pad token weights are configured and mapped to model configurations.

### Cell 6: Custom Chat Template Formatting & Zero-Padding Manual Packing
- Map message history roles: `"system"` -> `"system"`, `"user"` -> `"user"`, `"assistant"` -> `"model"`, and `"tool"` -> `"user"`.
- Tool calls emitted inside model turns must be parsed generically using XML brackets:
  `<call name="{tool_name}">{arguments}</call>`
- Tool output turns must be mapped generically using matching XML attribute tags:
  `<response name="{tool_name}">\n{content}\n</response>`
- Apply the chat template to the mapped conversation list dynamically using `tokenizer.apply_chat_template(..., tokenize=False)`.
- Implement **Zero-Padding Manual Sequence Packing** to maximize DGX GPU performance. Tokenize the entire corpus into a continuous ID string separated by the token EOS, chunk the array into block lengths of `MAX_SEQ_LENGTH` (4096), and batch-decode back to text blocks. This ensures 100% token utilization during the forward pass.

### Cell 7: Add LoRA PEFT Adapters
- Initialize PEFT adapter weights using Unsloth's `FastLanguageModel.get_peft_model()` using our target attention projections, rank, alpha, and `use_gradient_checkpointing="unsloth"`.

### Cell 8: SFTTrainer Configuration
- Initialize `SFTTrainer` with a detailed `SFTConfig` block.
- Set bf16 to True if supported by target hardware, adamw_8bit optimizer, weight decay, learning rate, manual shuffling, checkpoints savings and logging configurations.

### Cell 9: Resume-Aware Training Loop
- Use HuggingFace's `get_last_checkpoint(trainer.args.output_dir)` to auto-detect if previous runs crashed.
- Execute training: if a last checkpoint is found, run `trainer.train(resume_from_checkpoint=True)`, else restart fresh.

### Cell 10, 11 & 12: Saving, Smoke Testing and Verification
- Save parameters using standard `model.save_pretrained(LORA_OUTPUT_DIR)` and `tokenizer.save_pretrained()`.
- Run an output smoke test. Load the saved weights, output generations using a text streamer, and perform a total cold reload from disk in a fresh model variable block to verify model portable configuration.
```

---

## Prompt 2: Rebuilding the DPO Notebook (`pubmed_dpo_training_medgemma.ipynb`)

### Instructions to the LLM:
Copy, paste, and run the following prompt in your senior-developer generative AI workflow to construct the advanced Phase 2 notebook.

```text
Act as a Principal Deep Learning Architect specializing in preference optimization (RLHF/DPO) and DGX Unified Memory caching architectures.

Your task is to write a highly detailed, clean Jupyter notebook for Phase 2 Direct Preference Optimization (DPO) of a 27B parameter MedGemma model. The DPO stage aligns the model to refuse queries when a question goes beyond evidence and to call tools only when semantically required (Temporal Cognitive Gating), while avoiding host memory memory crashes.

The script must contain the following steps and blocks:

### Cell 1: Blackwell Triton Check & Critical Process Memory Fraction Cap (Optional)
- Force `os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"` to patch Triton FlexAttention Blackwell sm_120 shared-memory overflows.
- Enable `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.5,max_split_size_mb:256"` to limit RAM fragmenting.
- **The DGX Spark Unified Memory Wall (Critical Component):** Under unified memory architectures, PyTorch gets confused, viewing all host unified RAM as active GPU room and hoarding memory blocks forever. Show how to create a dedicated optional step:
  - Create an optional, standalone cell utilizing `torch.cuda.set_per_process_memory_fraction(0.55, 0)`.
  - Explain to the operator that this creates a hard "VRAM cap" at 55% (~70 GB on a 128 GB space) which blocks PyTorch from infinite RAM consumption, forcing caching garbage collection similar to discrete physical memory cards.

### Cell 2: SFT constants and dataset paths
- Define DPO hyperparameters:
  `DPO_DATA_FILE` targeting `pubmed_oncologist_v2_tool_dpo_messages.jsonl`
  `DPO_MAX_PAIRS = 5000` (stratified dataset ceiling)
  `LEARNING_RATE = 5e-6` (alignment learning rate must be much lower than SFT to avoid collapse)
  `DPO_BETA = 0.1` (KL penalty target)
  `LOSS_TYPE = "sigmoid"`
- Set paths for SFT Phase 1 LoRA adapters (`SFT_LORA_PATH`) and final output targets.

### Cell 3: Stratified Preference Dataset Sampling
- Load raw DPO pairs from JSONL. Counter-analyze the source types.
- If pairs exceed `DPO_MAX_PAIRS` (5,000), perform **Proportional Stratified Sampling**. Group records by their source values first, draw proportionally from each source index using random sampling, and prune the list to cap exactly at 5,000 pairs. This maintains perfect representational class density.

### Cell 4: Unified PEFT Loading & Attention Optimization
- Read previous SFT LoRA adapters using `FastLanguageModel.from_pretrained`.
- Overwrite attention implementation to Flash Attention 2: `model.config._attn_implementation = "flash_attention_2"`.
- Ensure native Gemma pad boundaries are loaded (`pad_token_id = 0`, `<pad>`).

### Cell 5: Validate and Extract Common Turn Suffixes
- DPO pairs diverge at a common user turn: Chosen has tool-use messages, Rejected writes direct-text answers.
- Implement a helper `_common_prefix_len(chosen_list, rejected_list)`.
- Track prompt messages up to the shared prefix point. Tokenize and apply the chat template to prompt messages.
- Convert Chosen and Rejected suffixes into target training strings:
  - Tool calls matching: `<call name="{tool_name}">{arguments}</call>`
  - Responses matching: `<response name="{tool_name}">\n{content}\n</response>`
- Check sequence lengths. Exclude any pair exceeding `MAX_SEQ_LENGTH` (4096) to prevent context overflows.

### Cell 6: Trainer Configuration & GC Empty Cache Callback
- Set up PEFT LoRA in training mode using `FastLanguageModel.for_training(model)`.
- Configure `DPOTrainer` with `precompute_ref_log_probs=False`. This is mandatory; doing one-shot in-memory precomputation will dynamically OOM unified hardware devices.
- Define a custom `TrainerCallback` subclassing `TrainerCallback`:
  - Hook into `on_step_begin` and `on_step_end` events.
  - Force physical cleanups under both boundaries using `torch.cuda.empty_cache()` and `gc.collect()`. This actively sweeps raw blocks at batch iterations, ensuring DPO trains stably.

### Cell 7: Sharded, Resumable Persistent Reference LogProb Cache
- TRL's standard `precompute_ref_log_probs=True` runs fully in system memory. If the process crashes near the end, all computed states are permanently lost.
- Write a custom persistent cache loop:
  - Split training rows into static shards of size 64.
  - Cycle through shards: run `trainer.compute_ref_log_probs` on the loader iteration, gather metrics across devices, and save to shard files (`shard-000XXX0.pt`) containing a configuration hash fingerprint.
  - If a crash occurs, re-running the cell reads the folder manifest, skips finished shards, resumes exactly where it crashed, and saves the final precomputed column array directly back to HuggingFace (`save_to_disk`).

### Cell 8, 9 & 10: DPO Training Loop, Saving, & Honesty Refusal Evaluator
- Detect checkpoints and execute `trainer.train()`.
- Save adapter weights. Show how the model saved as a single consolidated adapter of SFT+DPO relative to base MedGemma, meaning vLLM can deploy it directly under a single `--lora-modules` tag.
- End with a custom medical evaluation suite tracking Grounding, Boundary Refusal (stating uncertainties over weak studies), and Self-Correction on challenges.
```
