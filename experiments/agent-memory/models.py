"""Pydantic and LanceDB schemas for the memory system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pyarrow as pa
from pydantic import BaseModel, Field

import config


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"


# -- LanceDB table schema (pyarrow) --

MEMORY_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), config.EMBEDDING_DIM)),
    pa.field("memory_type", pa.string()),
    pa.field("importance", pa.float32()),
    pa.field("topic", pa.string()),
    pa.field("entities", pa.string()),         # JSON-encoded list
    pa.field("source_text", pa.string()),
    pa.field("created_at", pa.string()),       # ISO 8601
    pa.field("last_accessed_at", pa.string()), # ISO 8601
    pa.field("access_count", pa.int32()),
    pa.field("consolidated_into", pa.string()),  # ID of summary memory, or ""
    pa.field("consolidated_from", pa.string()),  # JSON-encoded list of source IDs, or ""
])


# -- Pydantic models for application logic --

class MemoryRecord(BaseModel):
    id: str = Field(default="")
    text: str
    vector: list[float] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    topic: str = ""
    entities: list[str] = Field(default_factory=list)
    source_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    consolidated_into: str = ""
    consolidated_from: list[str] = Field(default_factory=list)

    def to_lance_dict(self) -> dict:
        """Convert to a flat dict suitable for LanceDB insertion."""
        import json
        return {
            "id": self.id,
            "text": self.text,
            "vector": self.vector,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "topic": self.topic,
            "entities": json.dumps(self.entities),
            "source_text": self.source_text,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "consolidated_into": self.consolidated_into,
            "consolidated_from": json.dumps(self.consolidated_from),
        }

    @classmethod
    def from_lance_row(cls, row: dict) -> "MemoryRecord":
        """Reconstruct from a LanceDB row dict."""
        import json
        entities = row.get("entities", "[]")
        if isinstance(entities, str):
            entities = json.loads(entities)
        consolidated_from = row.get("consolidated_from", "[]")
        if isinstance(consolidated_from, str):
            consolidated_from = json.loads(consolidated_from)
        return cls(
            id=row["id"],
            text=row["text"],
            vector=row.get("vector", []),
            memory_type=row.get("memory_type", "fact"),
            importance=row.get("importance", 0.5),
            topic=row.get("topic", ""),
            entities=entities,
            source_text=row.get("source_text", ""),
            created_at=row.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_accessed_at=row.get("last_accessed_at", datetime.now(timezone.utc).isoformat()),
            access_count=row.get("access_count", 0),
            consolidated_into=row.get("consolidated_into", ""),
            consolidated_from=consolidated_from,
        )


class SearchResult(BaseModel):
    memory: MemoryRecord
    raw_similarity: float
    recency_score: float
    importance_score: float
    combined_score: float


class ExtractedFact(BaseModel):
    text: str
    memory_type: MemoryType = MemoryType.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    topic: str = ""
    entities: list[str] = Field(default_factory=list)


class ConsolidationGroup(BaseModel):
    topic: str
    memory_ids: list[str]
    summary_text: str = ""


class EvalQuery(BaseModel):
    query: str
    relevant_texts: list[str]
    description: Optional[str] = None
