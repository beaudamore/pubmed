# Stage 04: Tool Protocol And Tool SFT

Status: blocked

Tool training starts only after one exact MedGemma tool-call serialization and
vLLM parser complete a round trip through standalone inference, direct vLLM,
and OpenWebUI.

Planned source: an audited ToolACE subset with oncology and general no-tool
replay. BFCL remains evaluation-only. The successful multimodal oncology adapter
is the parent; every continuation must save to a new output path.

Blocker: the configured Hermes parser and the observed Gemma serialization have
not demonstrated compatibility.
