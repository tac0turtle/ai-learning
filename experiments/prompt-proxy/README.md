# Prompt Proxy — LLMLingua Compression Proxy for Claude Code & Codex

Transparent HTTP proxy that compresses agentic conversations before forwarding to upstream APIs. Combines rule-based structural transforms (inspired by [context-compactor](https://github.com/wjessup/context-compactor)) with [LLMLingua-2](https://github.com/microsoft/LLMLingua) semantic compression.

## Architecture

```
Claude Code / Codex CLI
        |
        v
  Proxy (:4000)
    1. Zone split (frozen / middle / hot)
    2. Strip thinking blocks (global, except last)
    3. Compact tool results >500 chars (middle zone)
    4. Strip narration filler (middle zone)
    5. LLMLingua-2 on remaining text (middle zone)
        |
        v
  Upstream API (Anthropic / OpenAI)
```

### Zone model

Conversations are split into three zones based on turn boundaries:

- **Frozen prefix** (first N turns) — never modified. Preserves Anthropic prompt caching (cache breakpoints need byte-stable prefixes).
- **Hot window** (last M turns) — never modified. The model needs recent context for coherent continuation.
- **Middle zone** — everything between. This is where all compression fires.

### Compression pipeline

1. **Strip thinking** — Remove `thinking` blocks from all assistant messages except the last. These are 40-46% of agentic session tokens.
2. **Compact tool results** — Replace large tool results in middle zone with `[Compacted: N lines, M chars]` plus first-line preview. Error results preserved.
3. **Strip narration** — Remove short filler text ("Let me...", "Sure...", "Great...") from assistant messages.
4. **LLMLingua-2** — Semantic token-level compression via XLM-RoBERTa-large. Scores and drops low-information tokens from remaining text blocks.

## Run

```bash
cd experiments/prompt-proxy
uv run proxy.py
```

Model downloads on first run (~500MB). Subsequent starts load from HuggingFace cache.

## Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 claude
```

## Use with Codex

```bash
OPENAI_BASE_URL=http://localhost:4000 codex
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `PORT` | `4000` | Proxy listen port |
| `UPSTREAM_ANTHROPIC_URL` | `https://api.anthropic.com` | Anthropic API base URL |
| `UPSTREAM_OPENAI_URL` | `https://api.openai.com` | OpenAI API base URL |
| `COMPRESS` | `true` | Enable/disable LLMLingua compression |
| `LLMLINGUA_RATE` | `0.5` | Target compression rate (lower = more aggressive) |
| `LLMLINGUA_MIN_CHARS` | `200` | Min text block size for LLMLingua |
| `FROZEN_PREFIX_TURNS` | `2` | Turns to protect at start |
| `HOT_WINDOW_TURNS` | `4` | Turns to protect at end |
| `MIN_MESSAGES` | `6` | Min messages before compression kicks in |
| `TOOL_RESULT_COMPACT_THRESHOLD` | `500` | Tool result size threshold |

## Response headers

Every proxied response includes `x-compacted-*` headers:

- `x-compacted-original-tokens` / `x-compacted-compressed-tokens`
- `x-compacted-tokens-saved` / `x-compacted-reduction-pct`
- `x-compacted-transforms` — which transforms fired
- `x-compacted-structural-ms` / `x-compacted-llmlingua-ms`

## Endpoints

| Route | Description |
|---|---|
| `POST /v1/messages` | Anthropic proxy (Claude Code) |
| `POST /v1/chat/completions` | OpenAI proxy (Codex) |
| `GET /health` | Health check with config |
| `* /{path}` | Passthrough to Anthropic upstream |

## Project structure

```
prompt-proxy/
  proxy.py        — FastAPI proxy server, routing, upstream forwarding
  pipeline.py     — Compression orchestration (zone split + transforms + LLMLingua)
  transforms.py   — Pure rule-based transforms (strip_thinking, compact_tool_results, strip_narration)
  compressor.py   — LLMLingua-2 wrapper (model loading, compression API)
  pyproject.toml  — Dependencies
```

## Device

Auto-detects: MPS (Apple Silicon) > CUDA > CPU.
