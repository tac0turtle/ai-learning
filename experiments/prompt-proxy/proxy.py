"""
LLMLingua compression proxy for Claude Code and Codex.

Sits between the CLI and the upstream API, applying:
  1. Rule-based structural transforms (strip thinking, compact tool results)
  2. LLMLingua-2 semantic compression on middle-zone text

Supports both Anthropic (/v1/messages) and OpenAI (/v1/chat/completions).

Usage:
    uv run proxy.py

Then point your CLI:
    ANTHROPIC_BASE_URL=http://localhost:4000 claude
    OPENAI_BASE_URL=http://localhost:4000 codex
"""

from __future__ import annotations

import copy
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import compressor
from pipeline import CompressionStats, PipelineConfig, compress_anthropic, compress_openai

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "4000"))
UPSTREAM_ANTHROPIC = os.environ.get(
    "UPSTREAM_ANTHROPIC_URL", "https://api.anthropic.com"
)
UPSTREAM_OPENAI = os.environ.get("UPSTREAM_OPENAI_URL", "https://api.openai.com")

CFG = PipelineConfig(
    frozen_prefix_turns=int(os.environ.get("FROZEN_PREFIX_TURNS", "2")),
    hot_window_turns=int(os.environ.get("HOT_WINDOW_TURNS", "4")),
    min_messages=int(os.environ.get("MIN_MESSAGES", "6")),
    tool_result_threshold=int(os.environ.get("TOOL_RESULT_COMPACT_THRESHOLD", "500")),
    llmlingua_rate=float(os.environ.get("LLMLINGUA_RATE", "0.5")),
    llmlingua_min_chars=int(os.environ.get("LLMLINGUA_MIN_CHARS", "200")),
    llmlingua_enabled=os.environ.get("COMPRESS", "true").lower() == "true",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LLMLingua Compression Proxy")
http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))


def _stats_headers(stats: CompressionStats) -> dict[str, str]:
    return {
        "x-compacted-original-tokens": str(stats.original_tokens_est),
        "x-compacted-compressed-tokens": str(stats.compressed_tokens_est),
        "x-compacted-tokens-saved": str(stats.saved_tokens),
        "x-compacted-reduction-pct": str(stats.reduction_pct),
        "x-compacted-transforms": ",".join(stats.transforms_applied),
        "x-compacted-structural-ms": str(round(stats.structural_time_ms, 1)),
        "x-compacted-llmlingua-ms": str(round(stats.llmlingua_time_ms, 1)),
    }


def _log(prefix: str, stats: CompressionStats) -> None:
    print(
        f"[{prefix}] {stats.original_tokens_est} -> {stats.compressed_tokens_est} tokens "
        f"({stats.reduction_pct}% reduction, transforms: {stats.transforms_applied})"
    )


async def _stream_response(upstream_resp: httpx.Response):
    async for chunk in upstream_resp.aiter_bytes():
        yield chunk


def _forward_headers(request: Request, keys: tuple[str, ...]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in keys:
        val = request.headers.get(key)
        if val:
            headers[key] = val
    headers.setdefault("content-type", "application/json")
    return headers


async def _proxy_request(
    upstream_url: str,
    body: dict,
    forward_headers: dict[str, str],
    stats: CompressionStats,
) -> StreamingResponse:
    is_stream = body.get("stream", False)

    if is_stream:
        upstream_resp = await http_client.send(
            http_client.build_request(
                "POST", upstream_url, json=body, headers=forward_headers
            ),
            stream=True,
        )
        resp_headers = dict(upstream_resp.headers)
        resp_headers.update(_stats_headers(stats))
        return StreamingResponse(
            _stream_response(upstream_resp),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )

    upstream_resp = await http_client.post(
        upstream_url, json=body, headers=forward_headers
    )
    resp_headers = dict(upstream_resp.headers)
    resp_headers.update(_stats_headers(stats))
    return StreamingResponse(
        iter([upstream_resp.content]),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def proxy_anthropic(request: Request):
    """Anthropic /v1/messages -- used by Claude Code."""
    body = await request.json()
    compressed, stats = compress_anthropic(
        copy.deepcopy(body.get("messages", [])), CFG
    )
    body["messages"] = compressed
    _log("anthropic", stats)

    headers = _forward_headers(
        request,
        ("x-api-key", "anthropic-version", "anthropic-beta", "content-type", "authorization"),
    )
    return await _proxy_request(
        f"{UPSTREAM_ANTHROPIC}/v1/messages", body, headers, stats
    )


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    """OpenAI /v1/chat/completions -- used by Codex."""
    body = await request.json()
    compressed, stats = compress_openai(
        copy.deepcopy(body.get("messages", [])), CFG
    )
    body["messages"] = compressed
    _log("openai", stats)

    headers = _forward_headers(
        request, ("authorization", "content-type", "openai-organization")
    )
    return await _proxy_request(
        f"{UPSTREAM_OPENAI}/v1/chat/completions", body, headers, stats
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str):
    """Pass through all other requests unmodified."""
    body = await request.body()
    fwd = {
        k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")
    }
    upstream_resp = await http_client.request(
        request.method, f"{UPSTREAM_ANTHROPIC}/{path}", content=body, headers=fwd
    )
    return StreamingResponse(
        iter([upstream_resp.content]),
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type"),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llmlingua_enabled": CFG.llmlingua_enabled,
        "llmlingua_rate": CFG.llmlingua_rate,
        "frozen_turns": CFG.frozen_prefix_turns,
        "hot_turns": CFG.hot_window_turns,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Preloading LLMLingua-2 model...")
    compressor._get_compressor()
    print(f"Proxy on :{PORT}")
    print(f"  Anthropic upstream: {UPSTREAM_ANTHROPIC}")
    print(f"  OpenAI upstream:    {UPSTREAM_OPENAI}")
    print(f"  LLMLingua:          {'ON' if CFG.llmlingua_enabled else 'OFF'} (rate={CFG.llmlingua_rate})")
    print(f"  Zones:              frozen={CFG.frozen_prefix_turns}, hot={CFG.hot_window_turns}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
