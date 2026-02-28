# optimizoor

AI optimization experiments. Each experiment is self-contained with its own `pyproject.toml` and `uv` environment — `cd` into the directory and `uv run` to get going.

## Experiments

| Experiment | Description |
|---|---|
| [prompt-compression](experiments/prompt-compression/) | Local prompt compression with LLMLingua-2 (BERT-based, ~500MB). FastAPI + browser UI. |
| [prompt-proxy](experiments/prompt-proxy/) | Transparent compression proxy for Claude Code & Codex. Rule-based structural transforms + LLMLingua-2 semantic compression with frozen/middle/hot zone architecture. |
| [agent-memory](experiments/agent-memory/) | Persistent agent memory via RAG + LanceDB + sentence-transformers. Extraction-first writes, time-weighted retrieval, MMR diversity, consolidation. CLI + FastAPI UI. |
