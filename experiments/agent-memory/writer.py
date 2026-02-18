"""Memory write pipeline: text -> LLM extract -> embed -> dedup -> store."""

import numpy as np
from anthropic import Anthropic

import config
import embeddings
import store
from models import ExtractedFact, MemoryRecord
from prompts import EXTRACT_FACTS_SYSTEM, EXTRACT_FACTS_TOOL


def extract_facts(text: str) -> list[ExtractedFact]:
    """Use the LLM to extract discrete facts from conversation text."""
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=2048,
        system=EXTRACT_FACTS_SYSTEM,
        tools=[EXTRACT_FACTS_TOOL],
        messages=[
            {"role": "user", "content": f"Extract memories from this text:\n\n{text}"}
        ],
    )

    facts = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "store_memories":
            for mem in block.input.get("memories", []):
                facts.append(ExtractedFact(
                    text=mem["text"],
                    memory_type=mem.get("memory_type", "fact"),
                    importance=mem.get("importance", 0.5),
                    topic=mem.get("topic", ""),
                    entities=mem.get("entities", []),
                ))
    return facts


def _find_duplicates(
    new_vectors: np.ndarray,
    threshold: float = config.DEDUP_COSINE_THRESHOLD,
) -> set[int]:
    """Check new vectors against existing memories for duplicates.
    Returns set of indices into new_vectors that are duplicates."""
    existing = store.get_all()
    if not existing:
        return set()

    existing_vectors = np.array(
        [row["vector"] for row in existing], dtype=np.float32
    )

    duplicates = set()
    for i, vec in enumerate(new_vectors):
        # Cosine similarity (vectors are already L2-normalized)
        sims = existing_vectors @ vec
        if np.max(sims) > threshold:
            duplicates.add(i)
    return duplicates


def write(text: str) -> list[str]:
    """Full pipeline: extract facts, embed, dedup, store.
    Returns list of stored memory IDs."""
    facts = extract_facts(text)
    if not facts:
        return []

    # Embed all extracted facts
    texts = [f.text for f in facts]
    vectors = embeddings.encode(texts)

    # Dedup against existing memories
    dupe_indices = _find_duplicates(vectors)

    # Build records for non-duplicate facts
    records = []
    for i, fact in enumerate(facts):
        if i in dupe_indices:
            continue
        record = MemoryRecord(
            text=fact.text,
            vector=vectors[i].tolist(),
            memory_type=fact.memory_type,
            importance=fact.importance,
            topic=fact.topic,
            entities=fact.entities,
            source_text=text,
        )
        records.append(record)

    if not records:
        return []

    ids = store.add_batch(records)
    return ids
