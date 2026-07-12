# Roadmap: High-Value Datagen & SFT Alignment Improvements

After completing your current **MedGemma 27B** fine-tuning run, you will have a baseline gated tool-calling oncology model. To establish an incremental engineering feedback loop, we have outlined **3 high-value architectural improvements** for your next training pass.

By running these improvements on the next pass, you can conduct comparative testing using our evaluation suites to mathematically gauge the improvement in logic, reliability, and retrieval success.

---

## Improvement A: Multi-Document Cross-Referencing (RAG Synthesis Synthesis)

### 1. The Limitation of the Baseline Model
Currently, SFT questions and answer pairs are synthesized using a **single, isolated context window (paragraph)**. This is a standard single-hop RAG setup. While it teaches simple extraction, it does not challenge the model to synthesize or resolve conflicting, multi-source literature during tool-calling steps.

### 2. The Architectural Improvement
We will update the question generator in `pubmed_datagen_v2_jupyterlab.ipynb` (Section 5) to ingest **2 to 3 related (or intentionally conflicting) clinical study abstracts** simultaneously. The question generator will be prompted to:
- Generate target questions that **cannot** be answered by reading only one of the provided contexts.
- Write synthesis queries that require resolving contradicting clinical trial metrics, staging classifications, or drug efficacy guidelines across different publication years.

### 3. SFT/DPO Grounding
The subsequent SFT answers will have the model reason through:
`<think>Abstract A states chemotherapy response is 40% in stage III, but Abstract B (a newer 2025 trial) corrects this rate to 55%. I must reconcile these differences and report the newer consensus...</think>`

### 4. How to Measure Success at Testing
Compare identical prompts to see if the model has learned **multi-hop comparative synthesis** or if it blindly regurgitates the first fact it retrieves.

---

## Improvement B: Tool-Calling API Failure Fallback Paths in DPO

### 1. The Limitation of the Baseline Model
Currently, our tool-calling SFT, DPO chosen, and DPO rejected steps operate on **perfect tool runs**. The tool is called, and a beautifully populated clinical trial or guideline database chunk always returns. 

At runtime, real APIs fail. Search registries return `0 results`, guidelines servers return `500 errors`, or connection bottlenecks time out. Because our SFT/DPO run has never seen a failing tool response, the model will likely panic, hang, repeat queries, or hallucinate a successful response anyway.

### 2. The Architectural Improvement
We will updated our DPO generator script (`augment_tool_calling_data.py`) to synthesize a subset of **dynamic API failure cases** (representing ~15% of our tool-calling records).

The DPO preference pairs will be generated as follows:
* **Tool Input:** Model queries `clinical_trials_api` for a rare biomarker mutation.
* **Tool Response:** The mock system returns a failure payload: `{"status": "error", "message": "Zero active registry records found.", "code": 404}`
* **DPO Chosen Turn:**
  ```xml
  <think>The trial database returned 0 results. I cannot verify the new compound outcomes. I must gracefully fallback to baseline clinical consensus while reporting the search failure...</think>
  The guideline database search did not return active outcome records for this specific mutation. However, standard clinical practice is to utilize systemic chemotherapy fallback profiles...
  ```
* **DPO Rejected Turn:** The model ignores the failure payload and confidently hallucinates fictional search outcomes:
  ```text
  The database results show a 72% response rate based on the fictitious trial...
  ```

### 3. How to Measure Success at Testing
Intentionally feed the inference model failing mock API inputs at testing. Verify if it handles the boundary condition gracefully and falls back to traditional practice guidelines, or if it hallucinates fictional data.

---

## Improvement C: SFT Token-Level Loss Masking (TRL Data Collator Integration)

### 1. The Limitation of the Baseline Model
Currently, your SFT pipeline calculates loss across **every token** in the entire text sequence. This means the model is penalized and optimized on predicting:
* The user's prompts (which are written by our generator, not by the model).
* SFT-system guidelines and policy blocks.
* The original incorrect turns in our self-correction sequences (before the user pushes back).

This dilutes the learning capacity of the model, wasting gradient steps on memorizing the user prompt formatting rather than learning the clinician reasoning engine.

### 2. The Architectural Improvement
We will integrate TRL's native `DataCollatorForCompletionOnlyLM` inside SFT Cell 8 (`Trainer Setup`):

```python
from trl import DataCollatorForCompletionOnlyLM

# Define the indicator tokens where Assistant output begins
response_template = "<start_of_turn>model\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)
```

By introducing this collator inside the `SFTTrainer` initialization blocks, active backpropagation weights will assign a loss target of `-100` to all tokens preceding the assistant's response.

### 3. Expected Result
The model's gradients will only compute loss on the clinician's **`<think>` block reasoning patterns**, standard dynamic tool-calls, and final answers. This concentrates SFT updates only on target behaviors, dramatically speeding up convergence and outputting much cleaner clinical logical alignments.
