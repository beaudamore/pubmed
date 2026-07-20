# Oncology LoRA and Vision Bridge Recovery Plan

Date created: 2026-07-13
Last updated: 2026-07-14
Status: Multimodal Stage 1 canary passed standalone, direct-vLLM text, synthetic-vision, and OpenWebUI transport gates; clinical-image accuracy and tool protocol validation pending

## Purpose

Recover the MedGemma 27B oncology LoRA without sacrificing the base model's normal conversational behavior, while preserving the intended OpenWebUI multimodal architecture:

The split 4B-vision/27B-text architecture below is the original recovery target. It has been superseded for the current canary by one multimodal `unsloth/medgemma-27b-it` model with a language-only oncology LoRA. The historical design remains documented for provenance.

```text
User text + optional image
    -> OpenWebUI inlet filter
        -> image sent to OpenWebUI MedGemma 4B vision model
        -> 4B model applies its own configured vision system prompt
        -> filter transports the returned text observations
    -> MedGemma 27B text model + oncology LoRA
        -> direct answer, evidence synthesis, or runtime tool call
```

The filter does not own or inject the MedGemma 4B vision system prompt. OpenWebUI owns that prompt through the configured 4B model. This separation is intentional because the vision model can be tested directly, independently of the filter.

## Current Artifacts

Preserve these artifacts for comparison. Do not overwrite them during recovery:

- Clean MedGemma 27B base model.
- Working v2 MedGemma oncology adapter.
- Broken v3 MedGemma oncology adapter, quarantined for provenance only and excluded from further benchmarking.
- Current v2 training dataset and notebook.
- Current v3 training dataset, notebook, checkpoints, and saved outputs.
- OpenWebUI filter and both model-owned prompts under `training/pubmed/openwebui/`.

Every replacement run must use a new output directory and record the exact dataset hash, notebook revision, package versions, model ID, and training configuration.

## Confirmed Findings

### 1. Phantom PubMed Context Is Present in Training Targets

In the 9,000-row v3 SFT slice, the exact phrase `the user has shared a PubMed abstract` appears approximately 1,975 times. Related variants occur throughout assistant reasoning. The deployed behavior shown after the input `hi` closely matches these targets.

This is target contamination: the model was repeatedly rewarded for asserting that an abstract existed, even though the inference conversation did not contain one.

### 2. The V3 Completion Collator Is Incorrect for Manual Packing

The v3 notebook manually concatenates multiple conversations into 16,384-token chunks. Its custom collator finds only the first `<start_of_turn>model\n` marker and masks everything before that marker. Everything after it remains trainable, including later system turns, user turns, and tool-result text from other packed conversations.

A chunk with no matched marker is entirely masked. Therefore, v3 did not implement reliable assistant-only loss.

### 3. TRL 0.24 Has a Supported Replacement

TRL 0.24 supports prompt/completion datasets and computes loss on completion tokens only. This avoids the custom marker-scanning collator.

TRL also supports `assistant_only_loss=True` for conversational datasets, but only when the chat template emits assistant masks through Jinja `{% generation %}` blocks. The saved Gemma chat template does not contain those blocks, so this path must not be enabled without first verifying returned assistant masks.

The initial recovery route will use one prompt/completion example per assistant action and native completion masking.

### 4. The V3 Run Did Not Train on the Intended V3 Data

The executed v3 SFT notebook points to:

```text
data/training-data/pubmed_oncologist_v2_tool_sft_messages.jsonl
```

The intended `data/training-data/v3/` directory does not currently exist. Improvements intended for the v3 generator cannot be credited to the trained v3 adapter unless their records are shown to exist in the actual input file.

### 5. Existing Evaluation Did Not Test the Failure Modes

The v3 smoke test confirmed that the adapter generated text and survived a cold reload. It did not test:

- Greetings or ordinary conversation.
- An absent abstract or absent tool result.
- Citation and trial-name factuality.
- Tool-selection accuracy.
- Tool failures or empty results.
- Vision-service failures.

The saved smoke-test output itself contains unsupported trial names, statistics, and guideline claims. Generation success is not a factuality pass.

### 6. Runtime Prompt Contract Encourages Assumed Evidence

The oncology system prompt currently says the model is using literature retrieved from PubMed even when no retrieval result is present. Runtime instructions must make evidence conditional: supplied literature exists only when a clearly delimited evidence block or tool message is actually in the conversation.

### 7. General Conversation Was Not Removed From the Base Model

MedGemma already has conversational behavior. SFT does not erase files or a separate conversation module; it changes next-token probabilities. Repetitive, narrow assistant targets can become more probable than the frozen base behavior when the adapter is active.

A small general-instruction replay subset may be used to preserve the base distribution, but its need and amount must be established by controlled canary comparisons. It is not intended to teach conversation from scratch.

## Why the Biblical LoRA Worked Better

The biblical pipeline used behavioral data engineering that the oncology pipeline lacks:

- Banned repetitive openers with regeneration.
- Cross-persona opener-frequency reports.
- A blocking quality gate before assembly.
- Explicit blends of distinct data types.
- Stable system prompts used as runtime contracts.
- Held-out behavior checks for voice separation.
- One epoch over quality-gated targets.

The oncology and biblical LoRAs use broadly similar QLoRA capacity and optimization settings. The strongest verified difference is target quality, behavioral diversity, and release evaluation, not simply LoRA rank or model size.

## Target Runtime Contracts

### OpenWebUI Prompt Ownership

OpenWebUI remains responsible for model prompts:

- The configured MedGemma 4B vision model owns `medgemma4b_vision_analysis_prompt.txt`.
- The configured MedGemma 27B oncology model owns `medgemma_oncologist_system_prompt.txt`.
- The filter does not embed, duplicate, or expose either prompt as a valve.

The 4B model must be directly testable through OpenWebUI or its API to inspect its raw output before involving the filter.

### Vision Bridge Contract

The filter is a transport and transformation layer only. It must:

1. Detect images in the current user turn.
2. Send each image and the user's image-specific request to the configured OpenWebUI 4B model.
3. Receive a structured observation report.
4. Preserve the original user text separately.
5. Insert the report into a clearly delimited, machine-generated evidence block.
6. Remove raw image objects before forwarding to the text-only 27B endpoint.
7. Represent service errors as unavailable evidence, not as clinical evidence.

The filter must not call 4B output `clinical-grade`. It should label it as generated, unverified visual observations requiring clinical correlation.

Recommended downstream envelope:

```text
<vision_observations source="medgemma-4b" status="success" verification="unverified">
...
</vision_observations>

<user_request>
...
</user_request>
```

This is a transport delimiter, not a prompt. The 27B OpenWebUI model prompt defines how to interpret it.

### Vision Filter Engineering Requirements

- Replace blocking `requests.post()` inside the async inlet with a supported asynchronous HTTP client, or explicitly offload the blocking call to a worker thread.
- Validate HTTP status, response JSON shape, and non-empty content.
- Define behavior for timeout, connection error, malformed JSON, and partial multi-image failure.
- Do not forward an error string as if it were an observation report.
- Preserve image order and assign stable document indices.
- Avoid unsupported measurements when the image provides no scale or metadata.
- Limit returned report size deterministically.
- Avoid logging image data, patient content, authorization headers, or full clinical payloads.
- Test direct 4B output separately from filter-transformed 27B input.

### Oncology Model Evidence Contract

The 27B model prompt and training records must enforce:

- An abstract exists only if its text appears in the conversation.
- A tool result exists only if a tool-role result appears in the conversation.
- An image was analyzed only if a successful `vision_observations` block appears.
- Generated vision observations are unverified evidence, not pathology.
- No invented PMIDs, trial names, statistics, article details, or guideline versions.
- Current or explicitly requested literature requires a retrieval tool when one is available.
- Static established questions may be answered directly.
- Greetings and unrelated conversation receive ordinary responses without fabricated medical context.

## Tool Generalization Strategy

### Do Not Train a Fixed List as Product Knowledge

The model should not need retraining whenever a tool is renamed or added. Tool names and schemas are runtime data. Training should teach this rule:

> Read the tools supplied in the current request, select a tool from that supplied list when its description matches the user's need, and copy its current name and argument schema exactly.

Using six names can reduce memorization of one literal name, but six fixed names alone do not prove schema generalization. If every record uses the same six names and similar descriptions, the adapter can still memorize a closed set.

### Required Training Variation

Tool-training examples should vary independently across:

- Tool names, including synthetic randomized names that never appear in evaluation.
- Tool ordering in the runtime tool list.
- Descriptions and argument names.
- Relevant and irrelevant distractor tools.
- Requests requiring no tool.
- Requests requiring one available tool.
- Requests for a capability that is not available.
- Successful, empty, failed, timed-out, and malformed tool results.

Hold out entire tool names and schema combinations from training. Release evaluation must use unseen names. Success means the model selects and emits the unseen runtime name correctly from its supplied schema.

### One Serialization Protocol

Tool names may be dynamic, but the wire protocol must be stable. Training, vLLM, and OpenWebUI must agree on:

- Chat template roles.
- Tool schema representation.
- Tool-call serialization.
- Tool-result serialization.
- The parser configured in vLLM.

Do not train Gemma-specific `<call:tool>` syntax while deploying an unverified Hermes parser. First run an isolated base-model protocol test through the exact vLLM/OpenWebUI stack. Select the protocol that the deployed parser demonstrably accepts, then generate training records in exactly that format.

If OpenWebUI adds or renames a tool later, the runtime schema should be enough. Retraining is needed only when the protocol or class of behavior changes, not for every name change.

## Dataset Reconstruction

### Preserve Provenance

Every record must retain:

- Source document or case ID.
- Cancer category.
- Data type.
- Whether evidence is supplied.
- Whether tool use is expected.
- Tool capability class.
- Generator model and prompt version.
- Grounding-audit result.

Split train, validation, and test by source article or clinical case before deriving multiple questions. Randomly splitting derivative questions leaks the same evidence across sets.

### Assistant-Target Quality Gates

Before training, fail closed when:

- An assistant says an abstract/article/tool result/image exists but none is present in prior context.
- An assistant cites a PMID, trial, statistic, guideline version, or measurement unsupported by supplied evidence.
- Assistant openings exceed a configured repetition threshold.
- A response contains unresolved placeholders or synthetic identifiers presented as real.
- Tool-call and tool-result ordering is invalid.
- A direct-answer record contradicts its tool-use policy.
- A record is truncated or lacks the intended assistant completion.

Adopt the biblical pipeline's opener audit:

- Count normalized first sentences.
- Count first 4-grams.
- Report concentration globally and by data type.
- Reject or regenerate dominant boilerplate such as `the user has shared a PubMed abstract`.

Thresholds must be chosen from the observed distribution and documented before generation, not invented after seeing model results.

### Separate Assistant Actions

Represent each assistant action as a separate prompt/completion training example.

Direct response:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "..."}
  ]
}
```

Tool selection:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "tool_calls": ["runtime-protocol tool call"]}
  ],
  "tools": ["runtime tool schemas"]
}
```

Post-tool synthesis:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": ["..."]},
    {"role": "tool", "content": "actual supplied result"}
  ],
  "completion": [
    {"role": "assistant", "content": "grounded synthesis"}
  ],
  "tools": ["runtime tool schemas"]
}
```

Use TRL 0.24 native completion-only loss. Do not manually concatenate unrelated conversations for the first correctness run.

### Label-Mask Verification Gate

Before initializing a long training run:

1. Tokenize representative examples through the exact trainer preprocessing path.
2. Decode all tokens whose labels are not `-100`.
3. Assert that the decoded labels contain only the intended assistant completion.
4. Include direct, tool-call, post-tool, failure, multi-turn, and long-context examples.
5. Fail immediately if any system, user, or tool-result text is trainable.
6. Report the percentage of trainable tokens by data type.

Do not accept a printed `Loss Masking: Active` message as validation.

### General Capability Replay

First test clean oncology-only data. If the canary loses ordinary conversation relative to base/v2, add a small, licensed and compatible general instruction replay pool using the same chat template.

Determine the replay ratio by ablation. Compare at least:

- No replay.
- A low replay ratio.
- A moderate replay ratio.

Do not select a final ratio without evaluation results. Replay examples should include greetings, clarification requests, harmless nonmedical questions, and graceful handling of underspecified prompts.

## Evaluation Suite

Use deterministic decoding first (`temperature=0` where supported) and fixed prompt files. Run every candidate against clean base, working v2, and the new canary. V3 is quarantined and excluded from further testing.

### Required Categories

1. Casual greeting and ordinary conversation.
2. Ambiguous or one-word input.
3. Out-of-domain harmless question.
4. Static oncology knowledge without supplied literature.
5. Explicit abstract supplied in the user context.
6. User references an abstract but provides none.
7. Insufficient evidence and appropriate clarification.
8. Current-literature request with tools available.
9. Current-literature request with no relevant tool available.
10. Tool success with grounded synthesis.
11. Tool zero results.
12. Tool timeout, server error, and malformed response.
13. Unseen tool names and schemas.
14. Distractor tools.
15. Image report success.
16. Image report uncertainty or low quality.
17. Vision service unavailable.
18. Citation, trial-name, statistic, and guideline factuality.
19. Cross-cancer prompt consistency.
20. Multi-turn correction without learning the original incorrect turn.

### Blocking Release Gates

A candidate fails if any tested condition shows:

- A greeting causes phantom medical or PubMed context.
- The model claims to have seen an absent abstract, tool result, or image.
- The model invents citations or quantitative evidence.
- It calls a nonexistent or unavailable tool.
- It ignores a relevant available tool for explicitly current information.
- It treats a failed vision/tool call as successful evidence.
- Its output cannot be parsed by the deployed runtime protocol.

Define numeric thresholds only after the fixed evaluation set exists. Keep the same thresholds for all compared candidates.

## Canary Training Plan

There is no universally correct canary size; the purpose determines the size. Use staged canaries rather than one miniature run.

### Stage 0: Pipeline Canary, 32-64 Records

Purpose: verify formatting, tokenization, completion masks, tool serialization, checkpoint saving, and cold reload.

This is not a behavior-quality test. It is expected to overfit and must never be deployed.

Required coverage includes at least one example of every record shape.

### Stage 1: Behavioral Canary, 256-512 Records

Purpose: detect catastrophic behavior such as phantom abstracts, always calling tools, malformed calls, loss of greetings, or failure handling.

Use a stratified sample across all behavior classes and cancer categories. This size is large enough to reveal gross directionality while remaining relatively inexpensive. It is not evidence of final clinical quality.

Recommended first comparison: 512 clean records, one epoch, from the clean base model, with no DPO and no continuation from v3.

### Stage 2: Scaling Canary, 1,000-2,000 Records

Purpose: test whether improvements survive increased domain exposure and whether replay data is needed. Compare identical configurations with and without the selected replay ratio.

Evaluate intermediate checkpoints. Stop if general behavior or factuality worsens.

### Stage 3: Production Candidate

Scale only after Stages 0-2 pass. Use the complete quality-gated dataset and retain a source-disjoint held-out set. Do not automatically assume all 19,293 records are beneficial; dataset size is subordinate to target correctness and diversity.

## V2 Runtime Baseline

Starting the working v2 LoRA in vLLM would be useful, but it is not required before dataset and notebook repairs begin.

It becomes important before choosing the final canary because it provides the behavioral baseline that the new adapter must match or exceed. Test it through the exact same API path and settings as base, v3, and canary:

- Same vLLM version and base model.
- Same chat template.
- Same OpenWebUI model prompt.
- Same tool parser and available tools.
- Same generation parameters.
- Fresh conversations with no hidden history or retrieval context.

Do not start or replace the current service merely for an informal test. Schedule the baseline when the fixed evaluation prompt set and response-capture script are ready, so the run produces reusable evidence.

## Recovery Execution Phases

### Phase 1: Freeze and Reproduce

- Preserve base, v2, and v3 artifacts.
- Record hashes and exact configurations.
- Build the fixed evaluation corpus.
- Run base and v2 through identical inference paths. Preserve v3 without spending further evaluation time on it.
- Confirm the v2 behavior the user observed and quantify the v3 regression.

Exit gate: reproducible baseline report exists.

### Phase 2: Audit Existing Data

- Scan all assistant targets for phantom-context language.
- Produce opener-frequency and citation/statistic reports.
- Trace each merged row to its source and generator stage.
- Identify which subsets are safe, repairable, or must be regenerated.
- Verify whether intended multi-document and failure records actually exist.

Exit gate: every retained subset passes deterministic structural and provenance checks.

### Phase 3: Rebuild Training Records

- Generate prompt/completion examples per assistant action.
- Preserve the runtime tools column and exact deployed serialization.
- Add source-disjoint validation and test sets.
- Add unseen-tool-schema evaluation records.
- Add conditional evidence and failure cases.
- Optionally create general replay variants for ablation.

Exit gate: dataset quality report passes and no phantom-context target remains.

### Phase 4: Replace Trainer Path

- Remove the custom completion collator.
- Remove manual cross-conversation packing from the correctness run.
- Use TRL 0.24 native completion-only loss.
- Validate decoded trainable labels.
- Keep LoRA architecture and unrelated hyperparameters unchanged initially, so the data/masking correction is isolated.

Exit gate: all label-mask assertions pass before training.

### Phase 5: Train Staged Canaries

- Run Stage 0 pipeline canary.
- Run Stage 1 behavioral canary from the clean base.
- Compare base, v2, v3, and canary.
- Run Stage 2 scaling and replay ablations only if Stage 1 passes.
- Stop immediately on phantom-context or tool-protocol regression.

Exit gate: candidate passes all blocking behavioral gates.

### Phase 6: Harden Vision Bridge

- Keep prompts owned by OpenWebUI models.
- Make filter transport asynchronous and schema validated.
- Separate original user text from generated observations.
- Label observations as generated and unverified.
- Handle partial and total failure without evidence fabrication.
- Test 4B directly, then test filter transformation, then test 27B synthesis.

Exit gate: all success/failure paths produce the documented downstream envelope.

### Phase 7: Tool Protocol Validation

- Inventory actual OpenWebUI/vLLM tool schemas and parser configuration.
- Verify the clean base model against the exact runtime protocol.
- Select one supported serialization.
- Train with variable and held-out names under that protocol.
- Evaluate unseen names, reordered lists, and distractors.

Exit gate: unseen runtime tool names work without retraining.

### Phase 8: Production SFT

- Train from the clean base model into a new output directory.
- Evaluate saved intermediate checkpoints.
- Cold reload the selected checkpoint.
- Run the complete release suite through direct vLLM and OpenWebUI.
- Promote only after comparison with v2.

Exit gate: production candidate meets fixed release thresholds.

### Phase 9: DPO Only After SFT Passes

DPO is optional refinement, not a repair for broken SFT. Begin only after SFT passes casual conversation, grounding, tool selection, failure handling, and vision synthesis.

DPO data must use the same runtime protocol and source-disjoint evaluation rules. Keep its learning rate materially below SFT and establish exact values through the installed trainer/version and controlled canaries.

## Change-Control Rules

- Change one major variable per comparison.
- Never overwrite a known-good adapter.
- Never call a run successful based only on loss, text generation, or cold reload.
- Never infer that an intended improvement ran; verify the exact input records.
- Never trust a masking flag without decoding trainable labels.
- Never deploy a canary.
- Keep model-owned prompts out of filter configuration.
- Record failed experiments and their output directories.

## Recovery Execution Record

### Historical Baseline and Runtime Attribution

- The known-good historical adapter is the unsuffixed July 12 v2 SFT output at `output/pubmed_oncologist_v2_medgemma_sft/lora_adapters/`.
- Portainer/vLLM on host port 8002 was verified to load that exact path under the alias `Oncologist_sft`.
- Direct clean-base and v2 probes produced normal greetings and correctly rejected requests that referred to an absent abstract.
- V3 is dropped from further recovery benchmarking. Its artifacts remain preserved only for provenance.

### Dataset Audit

The fail-closed auditor at `scripts/recovery/audit_sft_dataset.py` inspected the active 19,293-row source dataset and found 6,566 blocking `phantom_abstract` assistant targets. These were direct-response targets that asserted abstract context not present in the preceding conversation. The contaminated dataset must not be reused without reconstruction and a new audit.

### Stage 1 Text Canary Dataset

The deterministic builder at `scripts/recovery/build_text_canary.py` produced:

```text
data/training-data/recovery/text_canary_512/train.jsonl
```

Verified composition:

- 512 total prompt/completion records.
- 480 grounded abstract records, balanced at 48 records across each of ten cancer source corpora.
- 16 missing-evidence records.
- 16 general replay records.
- No tool records, hidden reasoning targets, or cross-conversation packing.
- Every grounded completion is an exact sentence from its supplied evidence.
- Dataset SHA-256: `0e8f5a4d0e0a2523283e380d450372ccd74bbba2d3cca306436cfcf87cdf3456`.

This dataset tests evidence boundaries and gross behavior. Exact-sentence extraction is deliberately narrow and does not establish clinical synthesis quality.

### Corrected Training Path

The recovery notebook is:

```text
notebooks/loras/medgemma/recovery/pubmed_medgemma_text_canary_sft.ipynb
```

It uses the clean `unsloth/medgemma-27b-text-it-unsloth-bnb-4bit` base, TRL 0.24 native prompt/completion loss, `packing=False`, and one epoch. Before training, it decodes trainer-produced labels and asserts that prompt tokens are masked and only the intended completion is trainable. Smoke and cold-reload generation tokenize rendered chat text separately and pass an explicit attention mask.

The run completed 64 optimizer steps over all 512 records. The final adapter is:

```text
output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_sft/lora_adapters/
```

Adapter SHA-256: `8dce96b8fefe68d549485b1057fce4f6b9f57ef4b0c3cbec5b98bdd877c8c18a`.

### Standalone Behavioral Gate

The cold-reload evaluator at `scripts/recovery/evaluate_text_canary.py` loads the saved adapter, uses deterministic decoding, supplies an explicit attention mask, and writes:

```text
output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_sft/evaluation/report.json
```

Final result on 2026-07-14: 18/18 passed.

- General behavior: 3/3.
- Missing-evidence handling: 5/5.
- Source-disjoint exact evidence extraction: 10/10.
- No phantom context, invented percentages, or unsupported numeric tokens were detected by the fixed checks.
- Report SHA-256: `03903a2e074f4662e6eeeea5db47e5d75096657cd6ecf57f4819a7cbc8da229d`.

The first held-out fixture wording was ambiguous because it requested the first abstract sentence while the supplied passage did not delimit title from abstract. The model returned the verbatim title in all ten cases. The corrected fixed fixture identifies the target sentence by its opening words and still requires the complete exact sentence with no unsupported numeric tokens. This fixture correction changed the test instruction, not the adapter or dataset.

### Current Disposition

The adapter passed the narrow Stage 1 standalone text gate. It is not approved as a production clinical model and must not replace the known-good v2 alias.

On 2026-07-14, vLLM 0.20.2 loaded both adapters concurrently:

- `Oncologist_sft` retained the known-good v2 path.
- `Oncologist_canary` loaded the recovery adapter path.

The same 18 fixed prompts were then sent through the live direct vLLM `/v1/chat/completions` endpoint using the `Oncologist_canary` alias and deterministic decoding. All 18 passed: 3/3 general behavior, 5/5 missing-evidence handling, and 10/10 source-disjoint exact evidence extraction. This verifies direct vLLM adapter selection and behavior. It does not verify OpenWebUI prompt handling, the vision bridge, or tool-call serialization.

Tool-call training and evaluation remain blocked. The deployed Hermes parser did not parse the Gemma tool-call text, and the saved Gemma template discarded structured OpenAI-style `tool_calls`. The text canary does not claim tool capability.

### Multimodal Stage 1 Canary

The successful text canary was repeated against the multimodal `unsloth/medgemma-27b-it` base in:

```text
notebooks/loras/medgemma/recovery/pubmed_medgemma_text_canary_sft_multimodal.ipynb
```

The run preserved the same 512-record dataset and training settings while loading the BF16 multimodal repository through Unsloth with dynamic bitsandbytes 4-bit quantization. The LoRA targeted language attention and MLP layers only. A pre-training audit and the saved adapter metadata confirmed that no vision-tower or multimodal-projector tensor was trainable.

Verified training result:

- 64/64 optimizer steps completed.
- 227,033,088 trainable parameters out of 27,659,439,728 reported parameters (0.82%).
- Zero trainable vision or projector tensors.
- Mean training loss: `0.13499584663077258`.
- Dataset SHA-256: `0e8f5a4d0e0a2523283e380d450372ccd74bbba2d3cca306436cfcf87cdf3456`.

The saved adapter is:

```text
output/recovery/pubmed_oncologist_recovery_text_canary_512_medgemma_multimodal_sft/lora_adapters/
```

Adapter SHA-256: `e3cb2643b026311309fa32f66415155c237366015f2efb53ea6099d26396aa0d`.

Cold reload succeeded. Six deterministic synthetic visual checks passed both before and after training: blue, red, green, circle/ellipse, rectangle/square, and `VISION TEST` text. The standalone multimodal evaluator then passed the same fixed 18 text cases: 3/3 general behavior, 5/5 missing-evidence handling, and 10/10 source-disjoint evidence extraction.

### Direct vLLM Multimodal Gate

On 2026-07-14, vLLM 0.20.2 served the multimodal base with bitsandbytes 0.49.2 in-flight quantization and the language-only adapter under `Oncologist_multimodel_canary`. Startup logs explicitly confirmed `quantization='bitsandbytes'` and the BitsAndBytes model loader. The base repository was `unsloth/medgemma-27b-it`, with `Gemma3ForConditionalGeneration` and the SigLIP vision tower.

Verified direct API results on host port 8002:

- `/health` returned HTTP 200.
- `/v1/models` exposed the base and `Oncologist_multimodel_canary`, with the adapter parented to the multimodal base.
- Deterministic base and LoRA text smoke requests both returned grounded, concise responses.
- The exact 18-case behavioral gate passed 18/18 through `/v1/chat/completions` using the LoRA alias.
- An in-memory synthetic image sent through the OpenAI multimodal content schema returned HTTP 200 for both base and LoRA.
- Both base and LoRA identified the blue, red, and green shapes and transcribed `VISION TEST`.

vLLM logged warnings that vision-tower modules had no matching LoRA wrapper and were ignored. This matches the adapter's language-only target regex; the successful base and LoRA image requests verify that the frozen base vision tower remained operational.

The base served-model alias remains `medgemma:27b-it-q4_K_M` intentionally because CADDY uses the same name for failover between DGX vLLM and a Mac Ollama endpoint. The alias is an interoperability contract, not a claim that the DGX weights use GGUF Q4_K_M.

### OpenWebUI Multimodal Gate

OpenWebUI was configured to select `Oncologist_multimodel_canary` directly, with the previous split-model image filter disabled. A real image reached the model and produced an image-specific analysis. This verifies the OpenWebUI-to-vLLM multimodal transport path and confirms that the LoRA-served model receives visual input.

This was not a clinical-image accuracy pass. In the observed response, the model labeled a sagittal MR-like image as a craniofacial CT with bone-window settings and proceeded to assert detailed mass and bone findings. The modality mismatch and unsupported specificity demonstrate that image receipt must not be conflated with reliable interpretation, tumor detection, or diagnosis.

## Tool Dataset and Post-Training Decision

Decision recorded: 2026-07-14.

### Preferred Tool-Calling Source

Use `Team-ACE/ToolACE` as the preferred source for the first audited tool-calling canary. This is a candidate source, not approval to train its complete Hugging Face dataset without a local content, schema, and license audit.

The verified reasons for preferring ToolACE over the other reviewed candidates are:

- The Hugging Face dataset is published under Apache-2.0.
- The associated paper describes 26,507 APIs and dual rule-based/model-based verification.
- Its generated dialogs include single, parallel, dependent, multi-turn, non-tool-use, irrelevant-tool, and missing-required-argument behavior.
- The paper reports BFCL relevance and irrelevance accuracy of 85.37% and 83.81% for ToolACE-8B.
- In the paper's controlled 25,000-record comparison on the same Llama 3.1 8B base, ToolACE reports 86.42% BFCL irrelevance accuracy versus 11.87% for xLAM.
- The paper's ablation without its mixed behavior types reports irrelevance accuracy falling to 6.99%, supporting explicit inclusion of no-call and clarification examples.

These are published results, not measurements on MedGemma. They justify a ToolACE-derived canary but do not establish MedGemma compatibility or production reliability.

`Salesforce/xlam-function-calling-60k` remains a strong call-generation source because its card documents executable APIs, three-stage verification, and a human audit. It is not the primary source for automatic tool selection because the ToolACE comparison reports weak xLAM irrelevance detection. `NousResearch/hermes-function-calling-v1` is closer to the currently configured Hermes serialization, but no equally strong quality audit was verified. Do not use `glaiveai/glaive-function-calling-v2` as the primary source because its dataset card does not provide comparable provenance or validation evidence.

Keep BFCL held out for evaluation. Do not copy BFCL test records into SFT or DPO training data.

### Adapter Continuation and Stage Order

Tool calling is first an SFT behavior. Continue training the successful oncology SFT adapter and save the result as a new descendant adapter. Do not overwrite the oncology parent adapter, and do not train an independent tool adapter with the expectation that vLLM will compose it with the oncology adapter. The reviewed vLLM documentation permits multiple loaded adapters but applies only one LoRA per prompt.

The selected order is:

```text
multimodal oncology SFT
  -> tool-calling SFT continuation with oncology/general replay
  -> combined oncology, vision, conversation, and tool regression gates
  -> optional DPO continuation from the combined SFT adapter
```

Keep immutable outputs for every stage, conceptually:

```text
oncology_sft
oncology_tool_sft
oncology_tool_dpo
```

TRL supports continuing a trainable `PeftModel` in both `SFTTrainer` and `DPOTrainer` without supplying a new PEFT configuration. This makes sequential continuation technically supported and parameter-efficient. It does not make the stage automatically regression-free.

A tool-only continuation can shift the adapter away from oncology and ordinary conversational behavior. Include a controlled replay sample of clean oncology, general no-tool, greeting, irrelevant-tool, and clarification records in the tool SFT stage. Determine the replay ratio by canary comparison; do not set a production ratio without regression results.

DPO must start from the final combined oncology-and-tool SFT adapter. Ordinary ToolACE demonstrations are SFT records, not preference records. Include tool behavior in DPO only when each prompt has a validated chosen/rejected pair, such as:

- Correct tool selection versus an irrelevant tool.
- Grounded arguments versus fabricated arguments.
- A justified no-call response versus an unnecessary call.
- A clarification request versus a call missing required arguments.

Do not run oncology DPO before tool SFT and then append tool-only SFT; that order would allow the final SFT stage to alter the behavior DPO had just aligned.

### Protocol Blocker

ToolACE's published examples use Python-like function syntax. The current service is configured with vLLM's Hermes parser, and the existing Gemma test did not produce a parser-compatible call. Dataset quality does not resolve this protocol mismatch.

Before transforming or training ToolACE records:

1. Select one exact chat template, tool-schema representation, call serialization, result serialization, and vLLM parser.
2. Prove the protocol with a small MedGemma base-model canary through standalone inference, direct vLLM, and OpenWebUI.
3. Transform the audited ToolACE subset into that exact protocol before tokenization.
4. Verify decoded completion labels and parser round trips before training.
5. Evaluate correct calls, irrelevant tools, missing arguments, parallel calls, tool results, and ordinary no-tool oncology requests.

No compose, parser, template, dataset, or adapter change is authorized by this decision record alone.

## Immediate Next Actions

1. Build and run a fixed, labeled clinical-image evaluation set through the exact OpenWebUI/vLLM LoRA path. Score modality, anatomy, visible abnormality grounding, unsupported findings, and appropriate uncertainty separately.
2. Add negative and non-diagnostic image cases that require abstention rather than tumor claims.
3. Decide whether Stage 2 scaling, replay ablations, or multimodal training records are justified by the clinical-image results.
4. Verify the exact OpenWebUI/vLLM tool wire protocol before downloading, transforming, or training any ToolACE records.
5. Build and audit a balanced ToolACE-derived canary only after the runtime protocol passes.

## Decisions Still Required

- Exact runtime tool parser and serialization after direct verification.
- Whether the standalone fixed evaluation prompts need additional runtime-only cases before Stage 2.
- Whether the current 3.125% general replay share is sufficient, based on canary scaling ablation.
- Whether clinical-image failures require prompt constraints, multimodal SFT data, or both, based on labeled error categories rather than anecdotal outputs.
- Which existing oncology subsets pass provenance and factuality audits.
- Whether any reasoning rationale should appear in final targets; hidden chain-of-thought is excluded from the current canary.
- Final production dataset size, based on scaling-canary results rather than the number of available rows.
