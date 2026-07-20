# YouTube Video Script: How We Fine-Tuned MedGemma 27B for Clinical Oncology Reasoning

---

## 📽️ VIDEO METADATA & PRODUCTION STYLE
*   **Video Type:** High-fidelity technical tutorial, system architecture breakdown, and live-code walk-through.
*   **Target Audience:** Machine learning engineers, clinical researchers, data scientists, and medical AI systems developers.
*   **Production Style:** Dual-layout. Alternates between high-resolution direct-to-camera speaking, professional architectural diagrams, and direct **VS Code / Jupyter / OpenWebUI screen shares**.
*   **Screen Share Setup:** Have two OpenWebUI model comparison windows ready side-by-side:
    *   **Left Pane:** Base `google/medgemma-27b-text-it` (or raw Qwen 14B V1 base model).
    *   **Right Pane:** Your custom **MedGemma 27B Token-Masked V3 Model** evaluating the exact EGFR Exon 19 Deletion lung adenocarcinoma clinical scenario shown in this script.

---

## 🎬 INTRODUCTION: Moving Beyond Chatbots

### [0:00 - 1:30]
**(Visual: High-quality camera, mid-shot. Host speaking directly. The screen behind holds a glowing terminal with training loss statistics steadily declining.)**

**HOST:**
"If you’ve ever tried to use a general-purpose LLM for complex clinical oncology decision-making, you know exactly where it breaks. You feed it a patient profile with a spiculated lung mass, an EGFR Exon 19 deletion, N2 mediastinal node involvement, and high PD-L1 expression. 

The general model gets excited about the high PD-L1 score and immediately recommends consolidation Immunotherapy—like Durvalumab. 

To any board-certified oncologist, that recommendation is a dangerous mistake. Immunotherapy is not only ineffective in EGFR-mutant cases, it can trigger severe, life-threatening pneumonitis when initiated right after radiation. The correct consensus choice is consolidating with Osimertinib—a targeted TKI—validated by the landmark LAURA Trial.

In this video, I’m going to show you how we built a highly specialized clinical reasoning engine to solve this. We are walk-stepping through how we transitioned a pipeline on our NVIDIA DGX from a lightweight Qwen 14B baseline to Google's massive **MedGemma 27B IT** model. 

We’ll look of the catastrophic failures along the way—including why our initial tool-calling models froze when APIs went down, and how the classic GPU memory tricks you use on discrete GPUs can actually choke your throughput on a Unified Memory architecture.

Let's look under the hood."

---

## 🎬 CHAPTER 1: The Transition from Qwen 14B to MedGemma 27B

### [1:30 - 3:00]
**(Visual: Split screen. Left side: The host speaking. Right side: Architectural schematic comparing Qwen-14B model footprint to MedGemma-27B. Under highlight is the "Medical Corpus Alignment" index.)**

**HOST:**
"Our journey started with a baseline fine-tuning run on a 14-billion parameter Qwen model. It was cheap, it was fast to train, and it gave us a solid starting pointer in OpenWebUI. 

But as we pushed the model into its custom clinical guideline alignment, we hit a hard performance ceiling. Wild-type oncology questions were handled fine, but the moment we tested multi-turn treatment reasoning across rare mutations or complex drug-drug resistance pathways, Qwen lacked the deep, localized factual knowledge to distinguish subtle clinical trial outcomes. 

We needed a base model with a native, pre-trained clinical vocabulary. That led us to Google’s **MedGemma 27B Text IT**. 

But jumping from 14B to a 27B model on our NVIDIA DGX with 128 gigabytes of Unified Memory presented major architectural challenges. Weight size in 4-bit was incredibly compact—about 16 gigabytes. But the active **activation memory overhead** required to backpropagate dense chain-of-thought `<think>` blocks through 27 billion parameters meant we had to systematically rewrite our sequence limits, dataset packing, and masking architectures.

Let's look at the first engineering wall we hit: context limits."

---

## 🎬 CHAPTER 2: Sequence Length & The SFT Token-Packing Math

### [3:00 - 5:00]
**(Visual: VS Code screen share. Highlight pubmed_sft_training_medgemma_v3.ipynb around MAX_SEQ_LENGTH = 16384 showing the manual sequence packing cell.)**

**HOST:**
"A standard text-only model defaults to a 4,096 context window. But inside our clinical datagen folder, we realized that single-hop contexts were holding us back. To teach the model deep synthesis, we upgraded our question generator to ingest 2 to 3 related (and sometimes intentionally conflicting) study abstracts simultaneously. 

That multi-hop abstract chunk represents over **2,000 tokens** alone. Add the patient’s clinical node history, the multi-step oncology reasoning, and the final guideline recommendation, and we easily exceed **5,000 to 6,000 tokens**. 

So, in SFT V3, we scaled our training context window to **16,384 tokens**. 

Now, when you do this, you might notice your step counts do something bizarre. During our baseline V2 SFT run with 1,000 examples, our script reported **96 total steps** and took about 2.5 hours. In our newly configured V3 run, with the *exact* same dataset size of 1,000, **the training steps dropped to exactly 24 steps!**

If you don't understand **Manual Sequence Packing**, you might panic and assume you lost 75% of your dataset. Here's what's actually happening under the hood:

To maximize GPU efficiency, we don't feed conversations one-by-one—that leaves massive padding gaps. Instead, we concatenate the entire tokenized dataset end-to-end with `<eos>` splitters and chop them into block-chunks of exactly our sequence length limit.

*   In V2, at `MAX_SEQ_LENGTH = 4096`, our token stream is chopped into **761 blocks**. Divided by an effective batch size of 8, we get **96 steps**.
*   In V3, at `MAX_SEQ_LENGTH = 16384`, our bucket is exactly **Four times wider**. The same token pool is chopped into only **190 blocks**. Divided by 8, we get **24 steps**!

So, no data was lost. One single step in V3 does the exact equivalent cognitive and backpropagation weight-updating work of four sequential steps in V2. It carries the same amount of water, just in a much larger bucket."

---

## 🎬 CHAPTER 3: The Unified Memory Fallacy on DGX Spark

### [5:00 - 6:30]
**(Visual: The host returns to direct-to-camera view. Behind him is the DGX performance dashboard showing System Memory at exactly 100.12 GB and GPU utilization pegged at 96%.)**

**HOST:**
"But carrying that bigger 16K bucket demands a massive physical toll. On our DGX Spark, our system memory jumped from ~60 gigabytes up to **100.12 gigabytes of active RAM**. 

Now, if you come from a standard x86 server setup with discrete PCIe GPUs and system CPUs, your immediate engineering instinct to prevent an OOM is to enable **DeepSpeed Activation Offloading** to move inactive tensors off the card and into CPU system RAM.

On unified memory architectures like the DGX, **this is a massive engineering trap.**

Because the CPU host cores and the GPU accelerator cores physically share the same unified pool of LPDDR5X, GPUMemory *is* CPU system RAM. 

If you instruct DeepSpeed to 'offload' activations, you aren't freeing up a single byte of physical space. You are simply forcing the memory controller to continuously shuffle pointers and copy memory blocks back and forth around the *same shared bus*, creating massive processing bottlenecks for zero net structural gain. 

If you offload to NVMe SSD disk files instead, you can bypass the 128GB wall, but your disk read/write throughput acts as a massive bottleneck, slowing training speeds by $10\times$ to $50\times$.

So on unified systems, our primary tools for context scaling must remain:
1.  **Unsloth Gradient Checkpointing** (recalculating intermediate activation states on-the-fly to save 80% RAM footprint).
2.  **TRL Token-Level Loss Masking**."

---

## 🎬 CHAPTER 4: Lessons Learned from Bad Training Runs

### [6:30 - 8:30]
**(Visual: Screen share in VS Code. Toggle between the legacy pubmed_sft_training_medgemma_v2.ipynb showing standard imports, and the new patched SFT v3 notebook highlighting the CustomCompletionCollator class.)**

**HOST:**
"Let's talk about the hard lessons we learned from failed training attempts. 

#### Fallback Failure 1: The 'API Fallback' Panic
In our initial tool-calling SFT runs, we trained the model on perfect scenarios. The model would write a tool call like `<call:clinical_trials_api>`, the database would always return a beautiful mock trial outcome, and the model would output its final answer. 

But at runtime, real APIs fail. Search servers return 500 errors, rate limits hit, or connection buffers time out. Because our initial model had only seen perfect runs, whenever we fed it an API failure payload, the model split-panicked, entered infinite loops, or completely fabricated a successful database response anyway.

To fix this, in our new version 3 dataset generator inside augment_tool_calling_data_v3.py, we synthetically injected **API failure states in 15% of our training runs**. 
*   **The Chosen turn** teaches the model to read the error (e.g., Error 504 Timeout) in its `<think>` block, report the lookup failure clearly with intellectual honesty, and gracefully fallback to standard clinical practice.
*   **The Rejected turn** penalizes the model via DPO if it tries to ignore the error and hallucinate mock data anyway.

#### Fallback Failure 2: The Deprecated Collator Trap
When we went to implement **Improvement C** (Token-Level Loss Masking), we tried importing `DataCollatorForCompletionOnlyLM` from TRL, which is the standard historical approach. 

But because our Jupyter container runs on **TRL v0.24.0**, we hit a complete block: **HuggingFace deprecated and completely removed `DataCollatorForCompletionOnlyLM` from the TRL imports interface!** They want developers using native dataset-level structures or tokenizer-level template masks. 

But when you pack sequences manually into flat, anonymous blocks of 16K, you lose the conversational turn metadata required to automatically generate these tokenizer masks. 

To bypass this without changing our packing algorithm, we wrote a **custom, inline subclass** directly inside the notebook: `CustomCompletionCollator(DataCollatorForLanguageModeling)`. It performs a sequential line-scan on batched labels, identifies our exact Gemma 3 model turn prefix `"<start_of_turn>model\n"`, and masks out all user and system tokens dynamically with a labels value of `-100`. 

This inline class solved our v0.24.0 import crash and keeps our memory completely aligned."

---

## 🎬 CHAPTER 5: Live Screen Share & Model Evaluation in OpenWebUI

### [8:30 - 10:00]
**(Visual: High-quality screen capture of OpenWebUI. Bring up the side-by-side prompt testing panel. Paste the exact Oncologist-to-AI query regarding the 58-year-old male with Stage IIIA EGFR-Mutant NSCLC.)**

**HOST:**
"Now, let’s jump into OpenWebUI for a live comparative evaluation. I’m pasting this highly realistic clinical scenario: a Stage IIIA lung adenocarcinoma patient, EGFR exon 19 deletion positive, N2 mediastinal nodes involved, with high PD-L1 TPS at 60%.

On the left pane, we have our legacy Qwen base model. On the right, we have our newly tuned **MedGemma 27B SFT v3** model with multi-hop RAG grounding and loss masking.

Let’s hit enter and watch the response generate."

**(Visual: Screen share watches both models stream in their output. Zoom in on the right pane's detailed <think> reasoning blocks.)**

**HOST:**
"Look at how the models diverge. 
The left model immediately recommends consolidation Durvalumab. It completely missed the biological conflict where the EGFR exon 19 mutation dominates therapeutic decision-making—making immunotherapy highly hazardous.

But look at our MedGemma V3 model on the right. 

Before it writes a single clinical recommendation, it launches an incredibly detailed, structured **`<think>` block**:
*   *It reasons through disease biology:* recognizing that Station 4R defines N2 mediastinal disease (Stage IIIA).
*   *It cross-references the LAURA Trial:* noting that while standard PACIFIC trials mandate Durvalumab consolidation, EGFR-mutant patients were excluded, and that the LAURA trial has proven a massive Progression-Free Survival (PFS) jump using Osimertinib instead (20.5 vs. 9.0 months, HR 0.16).
*   *It actively plans toxicity risks:* noting that initiating Osimertinib must be delayed by at least 4 weeks post-chemoradiation to prevent pneumonitis overlays.

Only *after* proving this logic does it deliver the final treatment summary, clinical caveats, and surveillance schedule fully aligned with NCCN Category 1 guidelines. 

This is the power of a highly focused clinical reasoning engine. 

---

## 🎬 OUTRO: The Multimodal Horizon

### [10:00 - 11:00]
**(Visual: Back to direct-to-camera, host mid-shot.)**

**HOST:**
"We are currently finalizing SFT v3 on our DGX node, and once SFT is established, we are diving straight into **DPO preference alignment** using our new tool-failure datasets.

Our next major milestone is **multimodal migration**—so the model can evaluate physical radiological slices, pathology slides, and DICOM images directly alongside text guidelines.

We have a fascinating design choice ahead of us. Do we train an end-to-end model on `google/medgemma-27b-it` and freeze the vision towers? Or do we build a modular, parallelizable pipeline where a lighter auxiliary VLM analyzes the raw CT scan, generates a highly detailed text description of the abnormality indexes, and passes that structured report to our primary text reasoner?

Let me know in the comments which architecture you would build. The full source notebooks and the technical switch guide are linked in the description below. 

Hit subscribe, and I will see you in the next one."

---
*(Visual: Outro screen showing social links, github repository, and subscribe button, accompanied by low-tempo diagnostic-style synth music.)*

---

## 📈 APPENDIX: EFFICACY ADVANTAGE & BUSINESS CASE (V3 vs. V2)

This appendix outlines the qualitative and quantitative improvements achieved during the transition from the SFT V2 baseline to the MedGemma 27B SFT V3 architecture.

### 1. Hard Business & Operational Benefits
*   **Mitigation of Clinical Liability & Risk:** V2 is prone to severe reasoning failure modes (such as recommending contraindicated immunotherapy, e.g., Durvalumab, for EGFR-mutant patients), presenting extreme translation risk. V3 guarantees strict alignment with oncology consensus guidelines, translating to a highly dependable and lower-liability enterprise tool.
*   **Production-Grade System Resilience:** Under standard production conditions, external APIs fail due to rate limits or connection timeouts. V3 was trained on synthetic API failures in 15% of its custom dataset, teaching the model to dynamically detect errors, gracefully communicate lookup constraints, and fallback to standardized clinical guidelines instead of looping or fabricating dummy data.
*   **Substantial Compute Cost Optimization:** Transitioning to **Manual Sequence Packing** with a context window of 16,384 tokens allows grouping multiple records end-to-end. This reduced GPU training steps from 96 down to 24 for 1,000 examples, significantly lowering active compute hours and maximizing training throughput.

### 2. Cognitive Architectural Enhancements (A Better "Brain")
*   **Expanded Semantic Window:** Expanding the training context limit to 16,384 tokens allows the model to simultaneously evaluate multiple clinical trial study abstracts, complete patient histories, and dual-diagnosed guidelines in-memory.
*   **Localized Clinical Vocabulary:** Selecting Google’s pre-trained **MedGemma 27B Text IT** base model provides native medical and pharmacological terminology that standard baselines cannot represent accurately.
*   **Loss-Masked Learning Efficiency:** Masking user and system prompts (`labels = -100`) via a custom completion collator ensures the model's backpropagation processes are 100% focused on weight updates inside the clinical `<think>` and reasoning blocks rather than prompt structure.

### 3. Improving Clinical Efficacy for Physicians and Researchers
*   **Elimination of Dangerous Clinical Artifacts:** V3 prioritizes dominant driver mutations (such as EGFR Exon 19 Deletions) over superficial markers like high PD-L1 expression, keeping clinicians from prescribing harmful pathways (such as immunotherapy prior to targeted TKI therapies like Osimertinib).
*   **Instant Grounded Evidence Synthesis:** Every clinical target is preceded by an in-depth, structured `<think>` trace that directly references and contrasts clinical data, such as citing the PACIFIC trial exclusions alongside the LAURA trial outcomes.
*   **Adverse Effect Planning:** V3 plans beyond basic medication recommendations, alerting the clinician to delay specific drug-initiation timings (e.g., waiting 4 weeks post-chemoradiation) to avoid toxic overlays like severe pneumonitis.
*   **Complex Multi-Abstract Grounding for Researchers:** For research questions involving obscure mutation overlays, the expanded context bucket ingests and synthesizes conflicting studies to resolve single-hop search limits.