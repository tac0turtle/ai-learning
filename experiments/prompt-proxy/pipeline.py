"""Compression pipeline: structural transforms + LLMLingua-2.

Orchestrates zone splitting, rule-based transforms, and semantic compression
into a single pass over the conversation messages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import compressor
from transforms import (
    compact_tool_results,
    strip_narration,
    strip_thinking,
    zone_split,
)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


@dataclass
class CompressionStats:
    original_tokens_est: int = 0
    compressed_tokens_est: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    llmlingua_time_ms: float = 0.0
    structural_time_ms: float = 0.0

    @property
    def saved_tokens(self) -> int:
        return self.original_tokens_est - self.compressed_tokens_est

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens_est == 0:
            return 0.0
        return round((self.saved_tokens / self.original_tokens_est) * 100, 1)


# ---------------------------------------------------------------------------
# LLMLingua pass on text blocks
# ---------------------------------------------------------------------------


def _compress_text_blocks(
    messages: list[dict],
    rate: float,
    min_chars: int,
) -> tuple[list[dict], float, int, int]:
    """Run LLMLingua-2 on text content blocks.

    Returns (compressed_messages, time_ms, original_tokens, compressed_tokens).
    """
    total_time = 0.0
    total_orig = 0
    total_comp = 0
    out = []

    for m in messages:
        content = m.get("content")

        # String content (OpenAI format)
        if isinstance(content, str) and len(content) >= min_chars:
            result = compressor.compress(text=content, rate=rate)
            total_time += result.time_ms
            total_orig += result.original_tokens
            total_comp += result.compressed_tokens
            out.append({**m, "content": result.compressed})
            continue

        # List content (Anthropic format)
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and len(block.get("text", "")) >= min_chars
                ):
                    result = compressor.compress(text=block["text"], rate=rate)
                    total_time += result.time_ms
                    total_orig += result.original_tokens
                    total_comp += result.compressed_tokens
                    new_blocks.append({**block, "text": result.compressed})
                else:
                    new_blocks.append(block)
            out.append({**m, "content": new_blocks})
            continue

        out.append(m)

    return out, total_time, total_orig, total_comp


# ---------------------------------------------------------------------------
# Full pipelines
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    frozen_prefix_turns: int = 2
    hot_window_turns: int = 4
    min_messages: int = 6
    tool_result_threshold: int = 500
    llmlingua_rate: float = 0.5
    llmlingua_min_chars: int = 200
    llmlingua_enabled: bool = True


def compress_anthropic(
    messages: list[dict], cfg: PipelineConfig = PipelineConfig()
) -> tuple[list[dict], CompressionStats]:
    """Full compression pipeline for Anthropic message format."""
    stats = CompressionStats()

    if len(messages) < cfg.min_messages:
        stats.original_tokens_est = _estimate_tokens(json.dumps(messages))
        stats.compressed_tokens_est = stats.original_tokens_est
        return messages, stats

    stats.original_tokens_est = _estimate_tokens(json.dumps(messages))

    # Phase 1: structural transforms
    t0 = time.perf_counter()

    messages, did = strip_thinking(messages)
    if did:
        stats.transforms_applied.append("strip_thinking")

    frozen, middle, hot = zone_split(
        messages, cfg.frozen_prefix_turns, cfg.hot_window_turns
    )

    if middle:
        middle, did = compact_tool_results(middle, cfg.tool_result_threshold)
        if did:
            stats.transforms_applied.append("compact_tool_results")

        middle, did = strip_narration(middle)
        if did:
            stats.transforms_applied.append("strip_narration")

    stats.structural_time_ms = (time.perf_counter() - t0) * 1000

    # Phase 2: LLMLingua semantic compression on middle zone
    if middle and cfg.llmlingua_enabled:
        middle, llm_time, orig_tok, comp_tok = _compress_text_blocks(
            middle, cfg.llmlingua_rate, cfg.llmlingua_min_chars
        )
        stats.llmlingua_time_ms = llm_time
        if orig_tok > comp_tok:
            stats.transforms_applied.append("llmlingua2")

    compressed = frozen + middle + hot
    stats.compressed_tokens_est = _estimate_tokens(json.dumps(compressed))
    return compressed, stats


def compress_openai(
    messages: list[dict], cfg: PipelineConfig = PipelineConfig()
) -> tuple[list[dict], CompressionStats]:
    """Full compression pipeline for OpenAI chat completion format."""
    stats = CompressionStats()

    if len(messages) < cfg.min_messages:
        stats.original_tokens_est = _estimate_tokens(json.dumps(messages))
        stats.compressed_tokens_est = stats.original_tokens_est
        return messages, stats

    stats.original_tokens_est = _estimate_tokens(json.dumps(messages))

    t0 = time.perf_counter()

    frozen, middle, hot = zone_split(
        messages, cfg.frozen_prefix_turns, cfg.hot_window_turns
    )

    if middle:
        middle, did = strip_narration(middle)
        if did:
            stats.transforms_applied.append("strip_narration")

    stats.structural_time_ms = (time.perf_counter() - t0) * 1000

    if middle and cfg.llmlingua_enabled:
        middle, llm_time, orig_tok, comp_tok = _compress_text_blocks(
            middle, cfg.llmlingua_rate, cfg.llmlingua_min_chars
        )
        stats.llmlingua_time_ms = llm_time
        if orig_tok > comp_tok:
            stats.transforms_applied.append("llmlingua2")

    compressed = frozen + middle + hot
    stats.compressed_tokens_est = _estimate_tokens(json.dumps(compressed))
    return compressed, stats
