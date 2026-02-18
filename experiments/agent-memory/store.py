"""LanceDB CRUD operations for the memory store."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa

import config
from models import MEMORY_SCHEMA, MemoryRecord

_db = None
_table = None


def _get_db():
    global _db
    if _db is None:
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(config.DB_PATH)
    return _db


def _get_table():
    global _table
    if _table is not None:
        return _table
    db = _get_db()
    if config.TABLE_NAME in db.table_names():
        _table = db.open_table(config.TABLE_NAME)
    else:
        empty = pa.table(
            {field.name: pa.array([], type=field.type) for field in MEMORY_SCHEMA},
            schema=MEMORY_SCHEMA,
        )
        _table = db.create_table(config.TABLE_NAME, data=empty)
    return _table


def reset_connection():
    """Reset cached connection (useful for testing)."""
    global _db, _table
    _db = None
    _table = None


def add(record: MemoryRecord) -> str:
    """Insert a memory record. Assigns an ID if empty. Returns the ID."""
    if not record.id:
        record.id = uuid.uuid4().hex[:12]
    table = _get_table()
    row = record.to_lance_dict()
    table.add([row])
    return record.id


def add_batch(records: list[MemoryRecord]) -> list[str]:
    """Insert multiple records in one call. Returns list of IDs."""
    if not records:
        return []
    for r in records:
        if not r.id:
            r.id = uuid.uuid4().hex[:12]
    table = _get_table()
    rows = [r.to_lance_dict() for r in records]
    table.add(rows)
    return [r.id for r in records]


def search_vector(query_vector: list[float], limit: int = 20) -> list[dict]:
    """ANN search returning top-limit results with _distance."""
    table = _get_table()
    results = (
        table.search(query_vector)
        .limit(limit)
        .to_list()
    )
    return results


def get_all(limit: int = 10000) -> list[dict]:
    """Return all rows (up to limit) as dicts."""
    table = _get_table()
    return table.search().limit(limit).to_list()


def get_by_id(memory_id: str) -> dict | None:
    """Fetch a single record by ID. Returns None if not found."""
    table = _get_table()
    results = table.search().where(f"id = '{memory_id}'").limit(1).to_list()
    if results:
        return results[0]
    return None


def update_access(memory_id: str):
    """Bump access_count and last_accessed_at for a memory."""
    table = _get_table()
    now = datetime.now(timezone.utc).isoformat()
    row = get_by_id(memory_id)
    if row is None:
        return
    new_count = row.get("access_count", 0) + 1
    table.update(
        where=f"id = '{memory_id}'",
        values={"access_count": new_count, "last_accessed_at": now},
    )


def mark_consolidated(memory_id: str, summary_id: str):
    """Mark a memory as consolidated into a summary."""
    table = _get_table()
    table.update(
        where=f"id = '{memory_id}'",
        values={"consolidated_into": summary_id},
    )


def count() -> int:
    """Return total number of memories."""
    table = _get_table()
    return table.count_rows()


def count_by_type() -> dict[str, int]:
    """Return counts grouped by memory_type."""
    rows = get_all()
    counts: dict[str, int] = {}
    for row in rows:
        mt = row.get("memory_type", "unknown")
        counts[mt] = counts.get(mt, 0) + 1
    return counts


def delete_all():
    """Drop and recreate the table. Use with caution."""
    db = _get_db()
    if config.TABLE_NAME in db.table_names():
        db.drop_table(config.TABLE_NAME)
    global _table
    _table = None
    _get_table()
