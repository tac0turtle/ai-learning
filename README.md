# optimizoor

AI optimization experiments. Each experiment is self-contained with its own `pyproject.toml` and `uv` environment — `cd` into the directory and `uv run` to get going.

## Experiments

| Experiment | Description |
|---|---|
| [prompt-compression](experiments/prompt-compression/) | Local prompt compression with LLMLingua-2 (BERT-based, ~500MB). FastAPI + browser UI. |
| [agent-memory](experiments/agent-memory/) | Persistent agent memory via RAG + LanceDB + sentence-transformers. Extraction-first writes, time-weighted retrieval, MMR diversity, consolidation. CLI + FastAPI UI. |
