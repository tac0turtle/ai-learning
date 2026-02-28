"""Rule-based structural transforms for agentic conversation compression.

Inspired by https://github.com/wjessup/context-compactor.
These are pure functions: messages in, messages out, no side effects.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Zone splitting (frozen / middle / hot)
# ---------------------------------------------------------------------------


def split_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Group messages into turns. A turn boundary is each user message."""
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(msg)
    if current:
        turns.append(current)
    return turns


def zone_split(
    messages: list[dict],
    frozen_turns: int = 2,
    hot_turns: int = 4,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split messages into frozen, middle, and hot zones.

    - Frozen prefix: never modified (cache stability).
    - Hot window: never modified (recent working memory).
    - Middle: compressible.
    """
    turns = split_into_turns(messages)
    total = len(turns)

    if total <= frozen_turns + hot_turns:
        return messages, [], []

    frozen_msgs = [m for t in turns[:frozen_turns] for m in t]
    hot_msgs = [m for t in turns[total - hot_turns :] for m in t]
    middle_msgs = [m for t in turns[frozen_turns : total - hot_turns] for m in t]
    return frozen_msgs, middle_msgs, hot_msgs


# ---------------------------------------------------------------------------
# Transform: strip thinking blocks
# ---------------------------------------------------------------------------


def strip_thinking(messages: list[dict]) -> tuple[list[dict], bool]:
    """Remove thinking blocks from all assistant messages except the last.

    Thinking blocks are 40-46% of agentic session tokens. The model doesn't
    need its old chain-of-thought once it has acted on it.
    """
    last_asst_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_asst_idx = i

    out = []
    applied = False
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or i == last_asst_idx:
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        filtered = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        ]
        if len(filtered) < len(content):
            applied = True
        if filtered:
            out.append({**m, "content": filtered})
    return out, applied


# ---------------------------------------------------------------------------
# Transform: compact large tool results
# ---------------------------------------------------------------------------

_ERROR_KEYWORDS = ("error", "exception", "traceback", "failed", "errno")


def compact_tool_results(
    messages: list[dict], threshold: int = 500
) -> tuple[list[dict], bool]:
    """Replace large tool_result content with a compact summary.

    Preserves error results intact -- those are high-signal.
    """
    out = []
    applied = False
    for m in messages:
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            out.append(m)
            continue

        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_content.append(block)
                continue

            result_content = block.get("content", "")
            if isinstance(result_content, list):
                text_parts = [
                    p.get("text", "")
                    for p in result_content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = "\n".join(text_parts)
            elif isinstance(result_content, str):
                text = result_content
            else:
                new_content.append(block)
                continue

            is_error = block.get("is_error", False) or any(
                kw in text.lower() for kw in _ERROR_KEYWORDS
            )
            if len(text) <= threshold or is_error:
                new_content.append(block)
                continue

            lines = text.splitlines()
            first_line = lines[0][:200] if lines else ""
            summary = f"[Compacted: {len(lines)} lines, {len(text)} chars]\n{first_line}"
            applied = True
            new_content.append({**block, "content": summary})

        out.append({**m, "content": new_content})
    return out, applied


# ---------------------------------------------------------------------------
# Transform: strip narration filler
# ---------------------------------------------------------------------------

_NARRATION_PREFIXES = (
    "let me",
    "i'll",
    "i will",
    "now i",
    "sure",
    "good",
    "great",
    "perfect",
    "done",
    "ok",
    "okay",
    "alright",
    "right",
    "here",
)


def strip_narration(messages: list[dict]) -> tuple[list[dict], bool]:
    """Remove short filler text from assistant messages.

    Short assistant text starting with narration prefixes ("Let me...",
    "Sure...", "Great...") carries near-zero information.
    """
    out = []
    applied = False
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, list):
            filtered = []
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and len(block.get("text", "")) <= 350
                    and block.get("text", "").lower().lstrip().startswith(
                        _NARRATION_PREFIXES
                    )
                ):
                    applied = True
                    continue
                filtered.append(block)
            if filtered:
                out.append({**m, "content": filtered})
        elif isinstance(content, str):
            if (
                len(content) <= 350
                and content.lower().lstrip().startswith(_NARRATION_PREFIXES)
            ):
                applied = True
                continue
            out.append(m)
        else:
            out.append(m)
    return out, applied
