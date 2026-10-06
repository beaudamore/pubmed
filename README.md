# PubMed Oncologist LoRA

> **Educational / Research Use Only** — not intended for clinical decision-making or patient care.

A **thinking-enabled oncology LoRA** that reasons through clinical oncology questions
step-by-step in `<think>…</think>` chains before answering, trained with Unsloth 4-bit QLoRA
in two stages (SFT, then DPO) on an NVIDIA DGX Spark. The training data is generated from
PubMed abstracts and CancerGUIDE patient cases with four anti-hallucination mechanisms:
grounding validation, boundary-awareness refusals, self-correction sequences, and
preference pairs that contrast grounded against fabricated answers.

Shipped twice: on **Qwen3-14B** (March 2026) and on **Qwen3.6-27B** (August 2026). The
Qwen3.6-27B adapter is published on Hugging Face as
[`beaudamore/pubmed-oncology-lora-qwen3.6-27b`](https://huggingface.co/beaudamore/pubmed-oncology-lora-qwen3.6-27b)
and has been served through vLLM `--enable-lora` behind Open WebUI, where two companion
tools keep a PubMed knowledge base growing:
[openwebui-pubmed-tool](https://github.com/beaudamore/openwebui-pubmed-tool) (in-chat deep
research) and [openwebui-feed-ingest-pipe](https://github.com/beaudamore/openwebui-feed-ingest-pipe)
(scheduled ingest with no model call).

Writeup: [damore.ai/blog/pubmed-oncology-fine-tuning-pipeline](https://www.damore.ai/blog/pubmed-oncology-fine-tuning-pipeline).

---

## Highlights

- **10 cancer types** — Bone, Brain, Breast, Colon, Gastric, Kidney, Lung, Ovarian, Prostate, Skin
- **Chain-of-thought reasoning** — every answer includes an explicit `<think>` reasoning chain before the conclusion
- **Anti-hallucination training** — grounding validation, boundary awareness ("beyond the evidence" refusals), and self-correction sequences
- **DPO alignment** — 31,651 preference pairs contrasting grounded vs. hallucinated answers
- **Treatment reasoning** from [CancerGUIDE](https://huggingface.co/datasets/microsoft/CancerGUIDE) synthetic patient cases
- **Abstract-free Q&A** — 8,000 questions answered without the source abstract in the prompt, so the adapter also answers from learned knowledge, not only from supplied context

## What it demonstrates

| Area | Specifics |
| --- | --- |
| **Synthetic data at scale** | ~$330 and 2–3 days of OpenRouter generation over 71K cleaned abstracts with a thinking teacher model; per-type, per-round, key-level resume so a crash costs minutes, not days. |
| **LLM-as-judge quality control** | Every answer judged Grounded / Extrapolated / Hallucinated against its source abstract; hallucinated answers become DPO rejections instead of being thrown away. |
| **Fine-tuning on a 27B multimodal base** | Qwen3.6-27B is a `*ForConditionalGeneration` model: vision tower frozen explicitly, adapter scope asserted, 16K-token packed sequences via a persistent token cache, DPO reference log-probs cached to disk and resumable. |
| **Responsible release** | Research-only model card, explicit out-of-scope uses, upstream license tracking (Apache 2.0, CC BY 4.0). |

## Data Sources

| Dataset | Source | License | Records |
| ------- | ------ | ------- | ------: |
| PubMed Cancer NLP | [cyberpsych/PubMed-Cancer-NLP-Textual-Dataset](https://huggingface.co/datasets/cyberpsych/PubMed-Cancer-NLP-Textual-Dataset) | Apache 2.0 | ~100K raw / 71K clean |
| CancerGUIDE | [microsoft/CancerGUIDE](https://huggingface.co/datasets/microsoft/CancerGUIDE) | CC BY 4.0 | 316 |

### PubMed Cancer NLP Dataset

- **Raw size:** 99,956 records across 10 CSVs
- **Clean size:** 71,343 records after cleaning
- **Format:** Title + Abstract from PubMed publications
- **Drop reasons (2026-03-04 run):** 19,406 short abstract, 8,131 duplicate, 882 no title, 162 retracted, 32 non-English

### CancerGUIDE

- **Size:** 316 records (165 structured + 151 unstructured)
- **Format:** Synthetic patient notes + treatment recommendations
- **Created by:** Microsoft Research using GPT-4.1

---

## Headline numbers

| Item | Value |
| --- | --- |
| **SFT set** | 40,020 ShareGPT conversations (`pubmed_oncologist_combined_sharegpt.jsonl`) |
| **SFT composition** | qa 14,007 · continuation 9,205 · qa_no_abstract 8,004 · treatment_reasoning 4,802 · self_correction 2,001 · beyond_evidence 2,001 |
| **DPO set** | 31,651 pairs: beyond_evidence 14,660 · grounding_reject 14,050 · self_correction 2,941 |
| **Datagen config (current)** | `MAX_RECORDS = 20,000` source records, `NUM_ROUNDS = 3` question angles × `QUESTIONS_PER_CHUNK = 3`, chunks of 1,500 chars on sentence boundaries |
| **Teacher model** | `qwen/qwen3-235b-a22b-thinking-2507` via OpenRouter |
| **Hardware** | NVIDIA DGX Spark (GB10, 128 GB unified memory) |

## Shipped adapters

Values read from `adapter_config.json` and `trainer_state.json`, not from the notebooks.
Adapters live under `output/` (gitignored).

| Adapter | Base | Date | Stage | Steps | Loss (first → last) | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| `pubmed_oncologist_v2_sft` | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | 2026-03-18 | SFT | 2,996 | 1.44 → 0.47 | r=32/α=32, seq 4096, manual packing |
| `pubmed_oncologist_v2_dpo_qwen3_14b` | same | 2026-03-30 | DPO | 741 | — | continues the SFT adapter |
| `pubmed_oncologist_v2_sft_qwen36_27b_unsloth_4bit` | `unsloth/Qwen3.6-27B` | 2026-08-07 | SFT | 241 | 1.11 → 0.58 | r=32, seq 16,384 packed, LR 2e-4, grad-accum 8, vision tower frozen |
| `pubmed_oncologist_v2_dpo_qwen36_27b_unsloth_4bit` | same | 2026-08-09 | DPO | 733 | — | 5,859 pairs after sampling and validation, β 0.05, LR 5e-6, seq 3072. **Published on Hugging Face.** |

Serving: the Qwen3.6 adapter ran as `oncologist_dpo_4bit_bnb` on a Qwen3.6-27B NVFP4 vLLM
stack (reference compose in the biblical repo's `compose/qwen_3_6_nvfp4-compose.yaml`). The
current production stack is Qwen3.8-27B and does not carry an oncology adapter; a Qwen3.8
retrain is the next training step.

## Architecture

| Component | Detail |
| --------- | ------ |
| **Datagen LLM** | Qwen3-235B-A22B Thinking (via OpenRouter) |
| **Judge LLM** | same model, structured Grounded / Extrapolated / Hallucinated verdicts |
| **Base models** | Qwen3-14B (first release), Qwen3.6-27B (current adapter), any Qwen3-family instruct model 14B+ works with the notebooks |
| **Hardware** | NVIDIA DGX Spark (128 GB unified memory) |
| **Training** | `unsloth-notebook` container (JupyterLab on host port 8889) |

### Thinking Model Details

Qwen3-235B-A22B is a Mixture-of-Experts model (235B total, 22B active parameters) that
**enforces reasoning** via `<think>...</think>` blocks. Every answer includes an explicit
reasoning chain before the conclusion. This is preserved in training data
(`KEEP_THINKING = True`) so the fine-tuned LoRA learns to reason through oncology problems.

---

## Project Structure

```text
pubmed/
├── .env.example              # OPENROUTER_API_KEY — copy to .env
├── README.md
├── hf/README.md              # Hugging Face model card for the Qwen3.6-27B adapter
├── scripts/
│   ├── clean_pubmed.py       # Download & clean source data
│   └── requirements.txt
├── notebooks/
│   ├── datagen/
│   │   ├── pubmed_datagen_sft.ipynb             # Q&A, quality gates, anti-hallucination sets, continuation, blend
│   │   ├── pubmed_datagen_dpo.ipynb             # Preference pairs from the SFT run's rejects and refusals
│   │   ├── pubmed_datagen_qa_no_abstract.ipynb  # Correction pass: adds the abstract-free Q&A category
│   │   └── pubmed_datagen.ipynb                 # Earlier single-notebook v2 (SFT + DPO in one), kept for reference
│   └── loras/
│       ├── pubmed_qwen3-14b-sft_training.ipynb  # Qwen3-14B SFT (March 2026)
│       ├── pubmed_dpo_training_v1.ipynb / _v2.ipynb   # Qwen3-14B DPO
│       └── qwen36/
│           ├── pubmed_qwen36-27b-sft_bnb4bit_dynamic.ipynb   # Qwen3.6-27B SFT (shipped run)
│           ├── pubmed_qwen36-27b-sft_bnb4bit.ipynb           # earlier variant
│           ├── pubmed_qwen36-27b-sft_NVFP4.ipynb             # experiment: train over the NVFP4 base
│           ├── pubmed_qwen36-27b-dpo_bnb4bit_v2.ipynb        # Qwen3.6-27B DPO (shipped run)
│           └── pubmed_qwen36-27b-dpo_nvfp4_v2.ipynb          # experiment
├── openwebui/
│   ├── filters/medgemma_vision_inlet_filter.py  # From the MedGemma experiments (see History)
│   └── prompts/                                 # Oncologist system prompt, vision analysis prompt
├── docs/                     # Design notes and the MedGemma-era reports (see History)
├── data/                     # (gitignored) generated by the pipeline
│   ├── source-raw/           # HF downloads
│   ├── source-clean/         # pubmed_{type}.jsonl, cancerguide_*.jsonl, cleaning_report.json
│   └── training-data/
│       ├── pubmed_oncologist_v2_dpo.jsonl       # Final DPO file (31,651 pairs)
│       └── pubmed_oncologist_v2/
│           ├── qa/ (+ _checkpoints/)            # Raw QA per cancer type, round tracking
│           ├── qa_validated/                    # Post-grounding-check QA
│           ├── qa_no_abstract/                  # Abstract-free QA + audit report
│           ├── cancerguide_reasoning/           # Treatment reasoning QA
│           ├── anti_hallucination/{beyond_evidence,self_correction}/
│           ├── augmented/continuation/          # Seed → completion pairs (no API)
│           ├── dpo/grounding_rejects/           # Hallucinated answers kept as rejections
│           ├── vertex-ai/                       # Gemma-format exports for the Vertex AI project
│           └── pubmed_oncologist_combined_sharegpt.jsonl   # Final SFT file (40,020)
└── output/<model_name>/{train,lora_adapters}/   # (gitignored)
```

---

## Quick Start

Notebooks run in JupyterLab inside the `unsloth-notebook` container (host port 8889), which
mounts this workspace at `/workspace/training`. Scripts run on the host.

```bash
# 1. Configure
cp .env.example .env            # add OPENROUTER_API_KEY
pip install -r scripts/requirements.txt

# 2. Clean source data (host, downloads from Hugging Face)
python scripts/clean_pubmed.py
#    -> data/source-clean/pubmed_{type}.jsonl, cancerguide_*.jsonl, cleaning_report.json

# 3. Generate training data (container; days and ~$330 at full scale, see "Estimated Scale")
#    notebooks/datagen/pubmed_datagen_sft.ipynb            -> pubmed_oncologist_combined_sharegpt.jsonl
#    notebooks/datagen/pubmed_datagen_dpo.ipynb            -> pubmed_oncologist_v2_dpo.jsonl
#    notebooks/datagen/pubmed_datagen_qa_no_abstract.ipynb -> adds qa_no_abstract/ and rewrites the combined file

# 4. Train (container). Check the GPU is free first:
docker ps                       # stop any vllm-* / sglang-* holding the GPU
#    notebooks/loras/qwen36/pubmed_qwen36-27b-sft_bnb4bit_dynamic.ipynb
#    notebooks/loras/qwen36/pubmed_qwen36-27b-dpo_bnb4bit_v2.ipynb

# 5. Audit the adapter for module-scope leakage before shipping it
docker exec unsloth-notebook python /workspace/training/docs/audit_adapters.py pubmed
```

`MODEL_NAME_BASE` in the SFT notebook is the contract with the DPO notebook, which resolves
the SFT adapter path from it. Change it in both places or not at all.

---

## Pipeline — Full Workflow

```text
PubMed + CancerGUIDE  ──►  Clean  ──►  SFT datagen  ──►  DPO datagen  ──►  SFT  ──►  DPO
       HuggingFace          │              │                  │              │          │
                        71K JSONL    QA + anti-halluc.   pairs from     LoRA       single
                                     + treatment         rejects and    adapter    SFT+DPO
                                     + continuation      refusals                  adapter
                                     + abstract-free QA
```

### Phase 1: Data Cleaning (`scripts/clean_pubmed.py`)

1. Download PubMed-Cancer-NLP dataset from HuggingFace
2. Normalize text (HTML unescape, Unicode NFC, ligature replacement, whitespace collapse)
3. Filter: min 200 chars abstract, min 10 chars title, max 15K chars, English-only (>85% Latin), retraction detection
4. SHA256 fingerprint deduplication
5. Write per-cancer-type JSONL files
6. Download CancerGUIDE (both `synthetic_structured` and `synthetic_unstructured` configs)
7. Clean and write CancerGUIDE JSONL files
8. Generate cleaning report

### Phase 2: SFT Data Generation (`notebooks/datagen/pubmed_datagen_sft.ipynb`)

<details>
<summary><b>Section 1-2: Configuration & Environment</b></summary>

- API config: OpenRouter endpoint + Qwen3-235B Thinking model
- `KEEP_THINKING = True` — preserve `<think>` blocks in training data
- Test mode: set `TEST_CHUNKS_PER_ROUND` to e.g. 20 for quick iteration (default 0 = full run)
- `MAX_RECORDS` — proportional random sample cap on source records before chunking (0 = use all)
- Dependencies: `openai`, `tqdm`, `nest_asyncio`, `tiktoken`, `pysbd`

</details>

<details>
<summary><b>Section 3: Load & Prepare Data</b></summary>

- Read cleaned JSONL files from `source-clean/`
- **Sentence-aware chunking** via pySBD + medical abbreviation protection
  - Text → sentences (handles et al., i.v., vs., mos., pts., Fig., etc.)
  - Sentences grouped into chunks up to `CHUNK_SIZE` chars
  - Overlap = last `OVERLAP_SENTENCES` (2) complete sentences carried forward
  - Every chunk starts and ends on a sentence boundary — no mid-word/mid-thought cuts
- Preview stats: records, chunks, avg chars per cancer type

</details>

<details>
<summary><b>Section 4: Oncologist Persona</b></summary>

- Single clinical oncologist system prompt
- Cancer-type specialization via `make_system_prompt(cancer_type)`
- Reasoning approach: mechanism → evidence → clinical application → limitations
- Rules: no fabricated statistics, explicit uncertainty acknowledgment

</details>

<details>
<summary><b>Section 5: Q/A Generation</b></summary>

- **Question rounds** (`NUM_ROUNDS`, currently 3): each round generates `QUESTIONS_PER_CHUNK`
  questions per chunk from a different angle — Mechanistic, Clinical Application, Critical Analysis
- Async with semaphore (`CONCURRENCY = 50`)
- 180s timeout per API call (thinking model needs time)
- `max_tokens = 4096` for answers (accommodates `<think>` chains)
- **Resume logic:** per cancer type, per round, and (since 2026-08-21) per key, with batched
  flushes, so an interrupted run resumes at the last written row
- Processes one cancer type at a time (memory bounded)

</details>

<details>
<summary><b>Section 5b: Quality Gate — Reasoning Depth</b></summary>

- `<think>` block presence rate (target: >80%)
- Answer length distribution
- Medical terminology density (30+ marker terms)
- Three-tier gate: PASS (≥80%), WARN (≥50%), FAIL (<50%)

</details>

<details>
<summary><b>Section 5c: Answer Grounding Check</b></summary>

- **LLM-as-judge:** validates each answer against its source abstract
- Three verdicts: **Grounded** / **Extrapolated** / **Hallucinated**
- String-based pre-filter: catches leaked source references ("according to the text", etc.)
- Strips `<think>` blocks before checking (judges the answer claims, not the reasoning)
- Writes validated per-type files to `qa_validated/`; hallucinated answers go to
  `dpo/grounding_rejects/` for the DPO stage
- Assembly uses validated files when available

</details>

<details>
<summary><b>Section 5d: "Beyond the Evidence" QA</b></summary>

- Generates questions the abstract **cannot** answer
- Model produces an honest refusal, explains the evidence gap, and describes what would be needed
- Question types: unstudied populations, missing long-term outcomes, unmentioned comparators
- Samples 30% of chunks, 2 questions each
- **The most important anti-hallucination mechanism** — teaches boundary awareness

</details>

<details>
<summary><b>Section 5e: Self-Correction Sequences</b></summary>

- Generates deliberate wrong answers → user pushback → corrected answers
- Wrong answer turn marked `"train": false` (masked during training)
- Model only learns the correction pattern, never the wrong answer
- Pushback templates: "Are you certain?", "That doesn't align with my understanding..."
- Samples 15% of validated QA
- 3 API calls per sequence (flawed + correction, plus the original question)

</details>

<details>
<summary><b>Section 6: CancerGUIDE Treatment Reasoning</b></summary>

- Direct treatment questions (3 per patient): "What treatment would you recommend?"
- What-if variants (2 per patient): altered comorbidities, mutations, age, metastasis sites
- Fill categories: CKD, CHF, diabetes, BRCA, MSI-H, EGFR, brain/liver/bone metastasis
- Grouped by patient for multi-turn conversations

</details>

<details>
<summary><b>Section 7: Assembly</b></summary>

- Read validated per-type files + CancerGUIDE reasoning
- Group QA by chunk → multi-turn ShareGPT conversations (`TURNS_PER_CONVERSATION = 4`)
- Quality filter: reject AI-speak and low-quality answers; dedupe on assembly

</details>

<details>
<summary><b>Section 8: Continuation Chunks</b></summary>

- Token-based chunking (`CONTINUATION_CHUNK_TOKENS = 500`)
- `split_seed_completion()`: ~60 token seed + ~440 token completion
- No API calls — raw abstract text teaches medical vocabulary and paper structure
- 5 instruction templates for variety

</details>

<details>
<summary><b>Section 9-10: Merge & Verify</b></summary>

- Loads all data sources and up/downsamples to the target blend ratios (table below), shuffles
- Format validation (system → human → gpt alternation), thinking-block rates by data type,
  cancer-type coverage per data type, sample conversations from each category

</details>

### Phase 2b: Abstract-Free QA (`notebooks/datagen/pubmed_datagen_qa_no_abstract.ipynb`)

A correction pass added on 2026-08-05, before the Qwen3.6 runs (the Qwen3-14B adapters were
trained on the earlier 33,349-row set without it). The original Q&A always carried the
source abstract in the user turn, which teaches the model to answer *from supplied context*
but gives no signal for answering *from knowledge*. This notebook audits the combined file,
generates and validates 8,000 abstract-free rows (the same kinds of questions, no abstract in
the prompt, answers still grounded in the source), writes `qa_no_abstract/` with an audit
report, rewrites the combined SFT file (keeping the previous one as `*.pre_qa_no_abstract.jsonl`),
and updates the SFT notebooks' category handling.

### Phase 2c: DPO Data Generation (`notebooks/datagen/pubmed_datagen_dpo.ipynb`)

Builds TRL-format preference pairs from what the SFT datagen already produced (details in
the [DPO section](#dpo--direct-preference-optimization)). Key-level resume and batched flush
for the grounding-reject and beyond-evidence sources (2026-08-21).

### Training Data Blend

| Data Type | Target % | Observed rows | Purpose |
| --------- | :------: | ------------: | ------- |
| Q/A (grounding-validated) | 42 | 14,007 | Core oncology knowledge with thinking chains |
| Continuation | 33 | 9,205 | Medical language patterns (no API calls) |
| Treatment reasoning | 15 | 4,802 | Clinical decision-making from patient cases |
| Beyond the evidence | 5 | 2,001 | Boundary awareness / honest refusal |
| Self-correction | 5 | 2,001 | Error recovery patterns |
| Abstract-free Q/A (added 2b) | — | 8,004 | Answering from learned knowledge |

### Phase 3: SFT Training (`notebooks/loras/qwen36/pubmed_qwen36-27b-sft_bnb4bit_dynamic.ipynb`)

Supervised fine-tuning via Unsloth 4-bit QLoRA on `unsloth/Qwen3.6-27B`:

- **Format:** ShareGPT rendered through the base model's chat template, `enable_thinking=False`
  at render time (the `<think>` content is in the data itself)
- **Input:** `pubmed_oncologist_combined_sharegpt.jsonl` (all 40,020 rows)
- **Multimodal base handling:** processor unwrapped to its tokenizer; `finetune_vision_layers=False`
  with an assertion that no vision or MTP module received an adapter; adapter audited again after reload
- **Packing:** a persistent `packed_token_cache/` (fingerprinted by input file and tokenizer) packs
  conversations into `MAX_SEQ_LENGTH = 16,384` token sequences once; re-runs reuse it
- **Hyperparameters:** LR 2e-4, r=32, grad-accum 8, 1 epoch, `use_gradient_checkpointing=True`
- **Durability:** `get_last_checkpoint` auto-resume, cold-reload verification, completion sentinel
- **Output:** `output/pubmed_oncologist_v2_sft_qwen36_27b_unsloth_4bit/lora_adapters/`

The Qwen3-14B notebook (`pubmed_qwen3-14b-sft_training.ipynb`) is the March 2026 original:
ChatML formatting, manual sequence packing into 4,096-token chunks, r=32/α=32, LR 2e-4,
batch 2 × 4 grad-accum, 1 epoch.

### Phase 4: DPO Training (`notebooks/loras/qwen36/pubmed_qwen36-27b-dpo_bnb4bit_v2.ipynb`)

Direct Preference Optimization that **continues training the SFT LoRA** (no new adapters stacked):

- **Input:** `pubmed_oncologist_v2_dpo.jsonl` (chosen/rejected pairs), `DPO_MAX_PAIRS = 9000`
  stratified by source; 5,859 pairs reached the trainer after the notebook's sampling and validation in the shipped run
- **Prerequisite:** the SFT adapter, resolved from `MODEL_NAME_BASE`; the inherited tower freeze is re-verified
- **Reference log-probs:** precomputed once into a persistent, fingerprinted `ref_logprobs_cache/`, resumable
- **Hyperparameters:** LR 5e-6 (~40× lower than SFT), β 0.05, sigmoid loss, `MAX_SEQ_LENGTH = 3072`, 1 epoch
- **Output:** a single combined SFT+DPO adapter ready for vLLM serving
- See [DPO Details](#dpo--direct-preference-optimization) below

---

## Anti-Hallucination Strategy

Four complementary mechanisms, inspired by [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit):

### Grounding Validation (Section 5c)

- **What:** LLM checks if every claim in the answer is traceable to the source abstract
- **How:** Thinking model as judge with structured verdict prompt
- **Result:** Hallucinated answers are rejected from SFT and reused as DPO rejections
- **Why it works:** Catches fabricated trial names, invented statistics, wrong mechanisms

### Boundary Awareness Training (Section 5d)

- **What:** Generate questions the abstract CAN'T answer → train with honest refusal responses
- **How:** Model explains what evidence is missing and what would be needed
- **Result:** LoRA learns to say "the available evidence doesn't address this"
- **Why it works:** Directly trains the skill of recognizing insufficient evidence

### Self-Correction Training (Section 5e)

- **What:** Wrong answer → user challenges → model corrects itself with reasoning
- **How:** Flawed answer is masked (`train=false`); model only learns the correction
- **Result:** LoRA learns to recover from errors rather than doubling down
- **Why it works:** Teaches error recognition and evidence-based recovery

### Source Reference Cleanup

- String-based filter rejects answers containing "according to the text", "the abstract states", etc.
- Hard-coded list of 12 leak patterns checked before grounding validation
- Conversations stand alone without referencing their source data

---

## Chunking Strategy

### Why Sentence-Aware Chunking Matters

| Stage | Bad chunk (mid-sentence cut) | Good chunk (complete sentences) |
| ----- | ---------------------------- | ------------------------------- |
| **Q/A generation** | LLM hallucinates missing context | Questions target complete ideas; answers grounded in full claims |
| **Grounding check** | Can't determine if claims are grounded when source is truncated | Clean source → accurate grounding verdicts |
| **Beyond-evidence** | Hard to know what "evidence" says when incomplete | Clear evidence boundary enables honest refusal |
| **Continuation** | Seed starts mid-thought; model learns broken patterns | Seed is a complete thought; model learns natural continuation |
| **Final LoRA** | Trains on fragments → choppy reasoning | Trains on coherent passages → complete clinical reasoning |

### Tool Choice: pySBD + Medical Abbreviation Protection

| Tool | Accuracy | Install Size | Speed |
| ---- | -------: | -----------: | ----: |
| Regex `rfind('. ')` | ~70% | 0 | Fastest |
| NLTK Punkt | ~40% | ~2 MB | Fast |
| pySBD (raw) | ~85% | 415 KB | 329 pass/sec |
| **pySBD + our medical abbrev layer** | **~94%** | **415 KB** | **329 pass/sec** |
| spaCy `en_core_web_sm` | ~90% | ~15 MB | ~100 pass/sec |
| scispaCy `en_core_sci_sm` | ~95% | ~100 MB | ~80 pass/sec |

**Decision:** pySBD + medical abbreviation protection gives 94% accuracy with 415 KB footprint, no ML dependencies, and 3x the throughput of spaCy. scispaCy would gain ~1% at 250x the install size.

<details>
<summary>Abbreviation categories protected (50+ terms)</summary>

- **Measurements:** approx., avg., mo., mos., yr., yrs., wk., hr., mg., ml., vs., no., vol., pt., pts.
- **Medical:** surg., adj., admin., diag., eval., sig., ther., resp., prog., prev.
- **Titles:** Dr., Prof., Mr., Mrs., Ms.
- **Academic:** et al., e.g., i.e., ed., eds., esp.
- **Routes:** i.v., i.m., s.c., p.o.
- **Genomic:** del., ins., mut., amp., chr.
- **Geographic:** U.S., E.U.

</details>

### Overlap: Complete Sentences, Not Raw Characters

**Previous approach (broken):** `CHUNK_OVERLAP = 200` characters — could start the next chunk with `"...tion was observed in 45% of patients with EGFR mutations."` — the LLM has no idea what "tion" refers to.

**Current approach:** `OVERLAP_SENTENCES = 2` — carries the last 2 complete sentences from the previous chunk. Every chunk starts with a complete thought, the LLM always has coherent context, and overlap is semantically meaningful.

---

## Output Format

All training data uses **ShareGPT multi-turn conversation format:**

```json
{
  "conversations": [
    {"from": "system", "value": "You are a clinical oncologist..."},
    {"from": "human", "value": "What mechanism drives..."},
    {"from": "gpt", "value": "<think>Let me analyze the pathway...</think>\n\nThe study demonstrates..."}
  ],
  "data_type": "qa",
  "cancer_type": "pubmed_lung_cancer"
}
```

| `data_type` | Turns | `<think>` blocks | Masked turns |
| ----------- | ----- | :--------------: | :----------: |
| `qa` | System + 2-8 (multi-turn) | Yes | No |
| `qa_no_abstract` | System + 2-8 | Yes | No |
| `treatment_reasoning` | System + 2-8 | Yes | No |
| `beyond_evidence` | System + 2 (single Q/A) | Yes | No |
| `self_correction` | System + 4 (Q → wrong → pushback → correct) | Yes (in correction) | Yes (`train=false` on wrong) |
| `continuation` | System + 2 (seed → completion) | No | No |

---

## DPO — Direct Preference Optimization

DPO is the **second training stage** after SFT. While SFT teaches the model *what to say*, DPO teaches it *what NOT to say* by contrasting good and bad responses to the same prompt.

**Training pipeline:** Base → SFT LoRA (Phase 3) → DPO continues training the same LoRA (Phase 4) → **single adapter**

### DPO Pair Sources

| Source | Chosen (Good) | Rejected (Bad) | Training Signal | Pairs |
| ------ | ------------- | -------------- | --------------- | ----: |
| **Grounding Rejects** | Re-generated strict-grounded answer (temp=0.2) | Hallucinated answer flagged by grounding check | Don't fabricate claims, statistics, or trial details | 14,050 |
| **Beyond-Evidence** | Honest refusal answer | Newly generated hallucinated answer (temp=0.9) | Know the limits of evidence; refuse gracefully | 14,660 |
| **Self-Correction** | Corrected answer with reasoning | Deliberately flawed answer | Fix mistakes when challenged; don't double down | 2,941 |

### DPO Data Format

TRL chat-template format compatible with `DPOTrainer`:

```json
{
  "chosen": [
    {"role": "system", "content": "You are a clinical oncologist..."},
    {"role": "user", "content": "What treatment would you recommend..."},
    {"role": "assistant", "content": "<think>...</think>\n\nBased on..."}
  ],
  "rejected": [
    {"role": "system", "content": "You are a clinical oncologist..."},
    {"role": "user", "content": "What treatment would you recommend..."},
    {"role": "assistant", "content": "The patient should receive..."}
  ],
  "source": "grounding_reject",
  "cancer_type": "pubmed_breast_cancer"
}
```

### DPO Hyperparameters (Qwen3.6-27B run)

| Parameter | Value | Rationale |
| --------- | ----- | --------- |
| `beta` | 0.05 | Gentle preference strength on a 27B base continuing an SFT adapter |
| `learning_rate` | 5e-6 | ~40× lower than SFT (2e-4) — fine adjustment, not major shift |
| `max_seq_length` | 3072 | Pairs include thinking blocks; bounded for the reference-logprob pass |
| `loss_type` | sigmoid | Standard DPO loss (Bradley-Terry model) |
| `epochs` | 1 | Single pass — DPO overfits quickly |
| Reference log-probs | cached to disk | Precomputed once, fingerprinted, resumable |

---

## Configuration Reference (SFT datagen, current values)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `CHUNK_SIZE` | 1500 chars | Soft char limit per chunk (never breaks mid-sentence) |
| `OVERLAP_SENTENCES` | 2 | Complete trailing sentences carried to next chunk |
| `MAX_RECORDS` | 20,000 | Proportional random sampling cap on source records before chunking (0 = use all) |
| `QUESTIONS_PER_CHUNK` | 3 | Questions per chunk per round |
| `NUM_ROUNDS` | 3 | Mechanistic, Clinical Application, Critical Analysis |
| `CONCURRENCY` | 50 | Max parallel API calls |
| `TEMPERATURE_QUESTIONS` | 0.8 | Higher for diversity |
| `TEMPERATURE_ANSWERS` | 0.4 | Lower for clinical accuracy |
| `TURNS_PER_CONVERSATION` | 4 | QA pairs per ShareGPT conversation |
| `CONTINUATION_CHUNK_TOKENS` | 500 | For continuation chunking |
| `CONTINUATION_SEED_TOKENS` | 60 | Seed size for continuation |
| `BEYOND_EVIDENCE_SAMPLE_FRACTION` | 0.30 | Fraction of chunks used |
| `SELF_CORRECTION_SAMPLE_FRACTION` | 0.15 | Fraction of QA sampled |
| `TEST_CHUNKS_PER_ROUND` | 0 | Set to >0 to limit chunks per round for testing |
| `DPO_MAX_PAIRS` (DPO notebook) | 9,000 | Stratified sample of DPO pairs for training |

---

## Resume & Checkpointing

The datagen notebooks are resume-safe:

- **Per cancer type:** if `{type}.jsonl` exists in `qa/`, that type is skipped
- **Per round:** round-tracking JSON in `qa/_checkpoints/` records how many chunks each round processed
- **Per key (since 2026-08-21):** Q&A generation, the grounding gate, and the DPO sources resume at
  the last written key with batched flushes
- **Per section:** each anti-hallucination section checks for existing output files
- **Continuation:** checks for non-empty per-type continuation files

To regenerate: delete the specific output file(s) and re-run the cell. Training notebooks
resume from `checkpoint-*` automatically and keep their packed-token and reference-logprob
caches across runs.

---

## Estimated Scale

### Full Corpus (`MAX_RECORDS = 0`, `NUM_ROUNDS = 1`, `QUESTIONS_PER_CHUNK = 3`)

- 71K cleaned records → ~47K chunks × 3 Q × 1 round = ~141K QA pairs
- ~14K beyond-evidence QA (30% sample)
- ~21K self-correction sequences (15% sample)
- ~47K continuation chunks (no API calls)
- DPO preference pairs from grounding rejects + beyond-evidence + self-correction

### Observed Runtime & Cost

| Phase | Hardware | Time | Cost |
| ----- | -------- | ---- | ---- |
| Datagen (full 71K records, 3 Q × 1 round) | API (Qwen3-235B via OpenRouter) | ~2–3 days | ~$330 |
| SFT, Qwen3-14B | NVIDIA DGX Spark (128 GB) | ~40 hrs | — |
| DPO, Qwen3-14B (`DPO_MAX_PAIRS = 9000`) | NVIDIA DGX Spark (128 GB) | ~14 hrs | — |

The current defaults (`MAX_RECORDS = 20,000`, three rounds) are tuned for a practical run that
produces a quality LoRA without the full-corpus cost. Full-scale runs (71K × 5 Q × 3 rounds)
are extremely API-intensive.

---

## History: experiments that are not the shipped pipeline

- **MedGemma 27B (July 2026).** A text SFT run on `unsloth/medgemma-27b-text-it` succeeded
  (`docs/medgemma_v3_sft_tuning_success_report_2026-07-13.md`, `docs/v3_slicing.md` for the
  incremental-slice training it used), followed by a multimodal recovery plan
  (`docs/oncology_lora_recovery_plan.md`, `docs/vision_base_model_switch.md`), a vision bridge
  Open WebUI filter (`openwebui/filters/`) and compose files (`docs/dgx-compose*.yaml`). The
  tool-calling work that came with it is written up in `docs/tool_calling_lessons_learned.md`.
  The MedGemma notebooks and the tool-calling augmentation scripts were removed from the repo
  on 2026-08-05 when the project returned to the Qwen line; the docs stay as the record.
- **Vertex AI export.** `data/training-data/pubmed_oncologist_v2/vertex-ai/` holds the SFT
  (33,349 rows) and DPO (31,670 rows) sets converted to the Gemma format used by the separate
  Vertex AI fine-tuning project.
- **Roadmap.** `docs/improvements.md` outlines three next-pass datagen improvements
  (multi-document synthesis, among others); `docs/datagen_notebook_and_prompts.md` and
  `docs/reverse_engineered_notebook_prompts.md` are blueprint-level writeups of the datagen.

---

## Dependencies

| Context | Packages |
| ------- | -------- |
| Cleaning script | `datasets`, `pandas`, `tqdm` |
| Datagen notebooks | `openai`, `tqdm`, `nest_asyncio`, `tiktoken`, `pysbd` |
| Training notebooks | `unsloth`, `torch`, `transformers`, `trl`, `peft`, `bitsandbytes` (installed in the container) |

---

## Known Gotchas

### OpenRouter Reasoning Field

OpenRouter returns thinking model reasoning in a **separate `reasoning` field** (`msg.model_extra['reasoning']`), NOT inline `<think>` tags in `msg.content`. The helper function `_extract_with_reasoning(resp)` recombines these into `<think>...</think>` format for training data. Without this fix, all generated answers lack thinking blocks (0% thinking rate). Applied to `generate_answer()`, `generate_beyond_evidence_answer()`, `generate_correction()`, and `generate_treatment_reasoning()`.

### Multimodal base

Qwen3.6-27B carries a vision tower and MTP heads. A bare `target_modules` list adapts them too,
and vLLM then refuses the adapter. The notebooks scope the adapter explicitly and assert it;
`../docs/audit_adapters.py` checks every saved adapter.

---

## Changelog

| Date | Change |
| ---- | ------ |
| 2026-03-04 | Project created: cleaning script, datagen notebook v1, anti-hallucination sections, sentence-aware chunking, OpenRouter reasoning-field fix, DPO pair collection |
| 2026-03-10 | Datagen v2 (JupyterLab), SFT notebook, round-0 checkpoints for all 10 cancer types |
| 2026-03-18 / 03-30 | Qwen3-14B SFT and DPO adapters trained |
| 2026-07-13 | MedGemma 27B text SFT run succeeded; multimodal recovery work begins |
| 2026-08-05 | MedGemma and tool-calling residue removed; project returns to the Qwen line. Abstract-free Q&A correction pass adds 8,000 rows to the SFT set (33,349 → 40,020) |
| 2026-08-07 / 08-09 | Qwen3.6-27B SFT and DPO adapters trained |
| 2026-08-16 | Hugging Face model card written; adapter published as `beaudamore/pubmed-oncology-lora-qwen3.6-27b` |
| 2026-08-21 | Datagen: `NUM_ROUNDS = 3`, `MAX_RECORDS = 20,000`, key-level resume and batched flush for Q&A, grounding gate and DPO sources; dedupe in assembly |
| 2026-08-25 | Vision/MTP tower scoping fixed across the Qwen3.6 notebooks |

## Related

- [openwebui-pubmed-tool](https://github.com/beaudamore/openwebui-pubmed-tool) — Open WebUI tool: PubMed search, PMID dedupe, per-article knowledge-base archiving with NLP entity extraction
- [openwebui-feed-ingest-pipe](https://github.com/beaudamore/openwebui-feed-ingest-pipe) — Open WebUI pipe that fills the same knowledge base on a schedule via Automations, with no LLM call
- [Hugging Face model card](https://huggingface.co/beaudamore/pubmed-oncology-lora-qwen3.6-27b) — the Qwen3.6-27B adapter (source in `hf/README.md`)
- [damore.ai: PubMed oncology fine-tuning pipeline](https://www.damore.ai/blog/pubmed-oncology-fine-tuning-pipeline) and [PubMed deep research tool](https://www.damore.ai/blog/pubmed-deep-research-tool)

## License

Training pipeline code: **MIT**

Training data sources retain their original licenses:

- PubMed Cancer NLP — Apache 2.0
- CancerGUIDE — CC BY 4.0

## Disclaimer

This project is for **educational and research purposes only**. The generated LoRA is not intended for direct clinical decision-making or patient care. Always consult qualified healthcare professionals for medical decisions.
