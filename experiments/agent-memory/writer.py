"""Memory write pipeline: text -> LLM extract -> embed -> classify -> store."""

import logging

import numpy as np
from anthropic import Anthropic

import config
import embeddings
import store
from classifier import classify_against_existing
from models import ExtractedFact, MemoryRecord, MemoryRelation
from prompts import EXTRACT_FACTS_SYSTEM, EXTRACT_FACTS_TOOL

logger = logging.getLogger(__name__)


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


def _build_record(
    fact: ExtractedFact,
    vector: list[float],
    source_text: str,
    relation: str = MemoryRelation.NEW.value,
    parent_id: str = "",
    root_id: str = "",
    version: int = 1,
) -> MemoryRecord:
    return MemoryRecord(
        text=fact.text,
        vector=vector,
        memory_type=fact.memory_type,
        importance=fact.importance,
        topic=fact.topic,
        entities=fact.entities,
        source_text=source_text,
        version=version,
        parent_id=parent_id,
        root_id=root_id,
        relation=relation,
        is_latest=True,
    )


def write(text: str) -> list[str]:
    """Full pipeline: extract facts, embed, classify against existing, store.

    Classification replaces the old cosine-threshold dedup. It uses either
    an NLI model or the LLM (configurable via CLASSIFIER_BACKEND env var)
    to determine: new, duplicate, updates, extends, or contradicts.

    Returns list of stored memory IDs.
    """
    facts = extract_facts(text)
    if not facts:
        return []

    # Embed all extracted facts
    texts = [f.text for f in facts]
    vectors = embeddings.encode(texts)

    stored_ids = []
    for i, fact in enumerate(facts):
        vec = vectors[i].tolist()

        # Classify against existing memories
        result = classify_against_existing(fact.text, vec)

        if result.relation == MemoryRelation.DUPLICATE:
            logger.info("Skipping duplicate: %s (matches %s)", fact.text, result.existing_memory_id)
            continue

        if result.relation in (MemoryRelation.UPDATES, MemoryRelation.CONTRADICTS):
            # Version chain: mark old as superseded, link new to old
            existing = store.get_by_id(result.existing_memory_id)
            if existing:
                store.mark_superseded(result.existing_memory_id)
                old_root = existing.get("root_id", "") or result.existing_memory_id
                old_version = existing.get("version", 1)
                record = _build_record(
                    fact, vec, text,
                    relation=result.relation.value,
                    parent_id=result.existing_memory_id,
                    root_id=old_root,
                    version=old_version + 1,
                )
                mid = store.add(record)
                logger.info(
                    "%s: %s -> %s (v%d, %s)",
                    result.relation.value, result.existing_memory_id, mid,
                    old_version + 1, result.reasoning,
                )
                stored_ids.append(mid)
                continue

        if result.relation == MemoryRelation.EXTENDS:
            # Store as new memory linked to the one it extends
            record = _build_record(
                fact, vec, text,
                relation=MemoryRelation.EXTENDS.value,
                parent_id=result.existing_memory_id,
            )
            mid = store.add(record)
            logger.info("Extends %s: %s (%s)", result.existing_memory_id, mid, result.reasoning)
            stored_ids.append(mid)
            continue

        # MemoryRelation.NEW — no existing memory matches
        record = _build_record(fact, vec, text)
        mid = store.add(record)
        logger.info("New memory: %s", mid)
        stored_ids.append(mid)

    return stored_ids
