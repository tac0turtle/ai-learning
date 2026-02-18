# Prompt Compression — LLMLingua-2

Local prompt compression using Microsoft's [LLMLingua-2](https://github.com/microsoft/LLMLingua) (BERT-based, ~500MB model). Reduces token count while preserving semantic meaning, useful for cutting LLM API costs and latency.

## Setup & Run

```bash
cd experiments/prompt-compression
uv run main.py
# http://localhost:8000
```

`uv run` creates the venv and installs deps automatically on first invocation.

The model downloads on first run (~500MB). Subsequent starts load from cache.

## API

### POST /api/compress

```json
{
  "prompt": "Your long prompt text here...",
  "rate": 0.33,
  "force_tokens": ["\n", ".", "?"]
}
```

Response:

```json
{
  "compressed": "Compressed text...",
  "original_tokens": 150,
  "compressed_tokens": 50,
  "ratio": 0.3333,
  "time_ms": 120.5
}
```

**Parameters:**
- `rate` (float, 0.05–0.95): Target compression rate. Lower = more aggressive. Default: 0.33.
- `force_tokens` (list[str]): Tokens preserved during compression. Default: `["\n", ".", "?"]`.

## Device

Auto-detects: MPS (Apple Silicon) > CUDA > CPU.
