"""Thin wrapper around LLMLingua-2 for prompt compression."""

import time
from dataclasses import dataclass

import torch
from llmlingua import PromptCompressor


@dataclass(frozen=True)
class CompressionResult:
    compressed: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    time_ms: float


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_instance: PromptCompressor | None = None


def _get_compressor() -> PromptCompressor:
    global _instance
    if _instance is None:
        device = _detect_device()
        print(f"Loading LLMLingua-2 model on {device}...")
        _instance = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map=device,
        )
        print("Model loaded.")
    return _instance


def compress(
    text: str,
    rate: float = 0.5,
    force_tokens: list[str] | None = None,
) -> CompressionResult:
    """Compress text using LLMLingua-2.

    Args:
        text: The text to compress.
        rate: Target compression rate (0.1 = aggressive, 0.9 = mild). Default 0.5.
        force_tokens: Tokens that must be preserved.

    Returns:
        CompressionResult with compressed text and stats.
    """
    if force_tokens is None:
        force_tokens = ["\n", ".", "?"]

    compressor = _get_compressor()
    start = time.perf_counter()
    result = compressor.compress_prompt(
        text,
        rate=rate,
        force_tokens=force_tokens,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    compressed_text: str = result["compressed_prompt"]
    original_tokens: int = result["origin_tokens"]
    compressed_tokens: int = result["compressed_tokens"]

    return CompressionResult(
        compressed=compressed_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        ratio=round(compressed_tokens / max(original_tokens, 1), 4),
        time_ms=round(elapsed_ms, 1),
    )
