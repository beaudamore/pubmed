# Oncology MedGemma 27B Multimodal V1

Status: active recovery and evaluation workstream
Started: 2026-07-13
Base model: `unsloth/medgemma-27b-it`
Current adapter: `Oncologist_multimodel_canary`

This directory is the navigation and provenance layer for the current oncology
MedGemma 27B multimodal work. Large data, notebooks, scripts, and model outputs
remain in the PubMed project's standard top-level directories so existing
container mounts, compose paths, and saved metadata remain valid.

## Stage Map

| Stage | State | Record |
| --- | --- | --- |
| 00 - Forensics and recovery design | Complete | [stages/00-forensics.md](stages/00-forensics.md) |
| 01 - Text canary | Complete | [stages/01-text-canary.md](stages/01-text-canary.md) |
| 02 - Multimodal canary | Complete | [stages/02-multimodal-canary.md](stages/02-multimodal-canary.md) |
| 03 - Clinical image evaluation | In progress | [stages/03-clinical-image-evaluation.md](stages/03-clinical-image-evaluation.md) |
| 04 - Tool protocol and tool SFT | Blocked | [stages/04-tool-protocol-and-sft.md](stages/04-tool-protocol-and-sft.md) |

## Source Of Truth

- Architecture and decisions: `docs/oncology_lora_recovery_plan.md`
- Machine-readable artifact inventory: `artifact-manifest.json`
- Training and evaluation data: `data/training-data/recovery/` and `data/evaluation/`
- Recovery scripts: `scripts/recovery/`
- Recovery notebooks: `notebooks/loras/medgemma/recovery/`
- Immutable run outputs: `output/recovery/`
- Runtime configuration: `docs/dgx-compose.yaml`

## Change Rules

1. Do not move or overwrite completed adapters, reports, datasets, or notebooks.
2. Give every new stage output a new path and record its hash in the manifest.
3. Pin external datasets to an immutable repository revision.
4. Keep benchmark test splits evaluation-only.
5. Record a stage as complete only after cold reload and runtime-path validation.
