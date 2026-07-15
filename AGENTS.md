# Project Rules

- DO NOT modify/add/delete any files/folder not in this directory. only be restricted with this directory.
- Use `uv` for dependency management, environment creation, commands, and notebooks.
- Keep the project simple: plain Python modules under `src/` are preferred over package scaffolding.
- Keep experiment code importable from notebooks; avoid one-off logic that cannot be reused.
- Keep RAG and CAG controlled by the same dataset subset, generator, prompt policy, decoding settings, and evaluation protocol.
- Treat offline preparation time separately from online latency.
- Do not include ACQAD decomposition fields in model prompts or evaluation records.
- Prefer deterministic defaults: fixed seed, explicit model identifiers, JSONL row outputs, and CSV summaries.
