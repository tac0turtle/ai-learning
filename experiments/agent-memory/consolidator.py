"""Memory consolidation: group old memories by topic, summarize, link provenance."""

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from anthropic import Anthropic

import config
import embeddings
import store
from models import MemoryRecord, MemoryType
from prompts import CONSOLIDATE_SYSTEM, CONSOLIDATE_TOOL


def _get_old_unconsolidated(age_days: int = config.CONSOLIDATION_AGE_DAYS) -> list[dict]:
    """Fetch memories older than age_days that haven't been consolidated."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)
    all_rows = store.get_all()
    return [
        row for row in all_rows
        if row.get("consolidated_into", "") == ""
        and row.get("memory_type", "") != "summary"
        and datetime.fromisoformat(row["created_at"]) < cutoff
    ]


def _group_by_topic(rows: list[dict]) -> dict[str, list[dict]]:
    """Group memories by their topic field."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        topic = row.get("topic", "general") or "general"
        groups[topic].append(row)
    return groups


def _summarize_batch(memories_text: list[str], topic: str) -> dict:
    """LLM call to summarize a batch of related memories."""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    formatted = "\n".join(f"- {t}" for t in memories_text)
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system=CONSOLIDATE_SYSTEM,
        tools=[CONSOLIDATE_TOOL],
        messages=[
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nMemories to consolidate:\n{formatted}",
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_summary":
            return block.input
    return {"summary_text": formatted, "importance": 0.5, "entities": []}


def consolidate() -> int:
    """Run consolidation on old memories. Returns number of summaries created."""
    old_rows = _get_old_unconsolidated()
    if not old_rows:
        return 0

    groups = _group_by_topic(old_rows)
    summaries_created = 0

    for topic, rows in groups.items():
        # Process in batches
        for batch_start in range(0, len(rows), config.CONSOLIDATION_BATCH_SIZE):
            batch = rows[batch_start:batch_start + config.CONSOLIDATION_BATCH_SIZE]
            if len(batch) < 2:
                continue

            texts = [r["text"] for r in batch]
            source_ids = [r["id"] for r in batch]

            result = _summarize_batch(texts, topic)

            # Create summary memory
            summary_vector = embeddings.encode_single(result["summary_text"])
            summary = MemoryRecord(
                text=result["summary_text"],
                vector=summary_vector,
                memory_type=MemoryType.SUMMARY,
                importance=result.get("importance", 0.5),
                topic=topic,
                entities=result.get("entities", []),
                source_text="",
                consolidated_from=source_ids,
            )
            summary_id = store.add(summary)

            # Mark source memories as consolidated
            for src_id in source_ids:
                store.mark_consolidated(src_id, summary_id)

            summaries_created += 1

    return summaries_created
