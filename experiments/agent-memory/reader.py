"""Time-weighted retrieval with MMR diversity re-ranking."""

import math
from datetime import datetime, timezone

import numpy as np

import config
import embeddings
import store
from models import MemoryRecord, SearchResult


def _recency_score(created_at: str | datetime) -> float:
    """Exponential decay based on age. Returns 0.0-1.0."""
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400
    half_life = config.RECENCY_HALF_LIFE_DAYS
    return math.exp(-0.693 * age_days / half_life)  # ln(2) ≈ 0.693


def _combined_score(similarity: float, recency: float, importance: float) -> float:
    """Weighted combination of the three signals."""
    return (
        config.WEIGHT_SIMILARITY * similarity
        + config.WEIGHT_RECENCY * recency
        + config.WEIGHT_IMPORTANCE * importance
    )


def _similarity_from_distance(distance: float) -> float:
    """Convert LanceDB L2 distance to cosine similarity.
    For normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)."""
    return max(0.0, 1.0 - (distance / 2.0))


def search(query: str, k: int = config.MMR_DEFAULT_K) -> list[SearchResult]:
    """Search memories with time-weighted scoring and MMR re-ranking."""
    query_vector = embeddings.encode_single(query)

    # Oversample candidates for MMR
    oversample = max(k, config.MMR_OVERSAMPLE)
    raw_results = store.search_vector(query_vector, limit=oversample)

    if not raw_results:
        return []

    # Score all candidates
    candidates = []
    for row in raw_results:
        # Skip consolidated memories (they've been merged into summaries)
        if row.get("consolidated_into", ""):
            continue

        distance = row.get("_distance", 0.0)
        sim = _similarity_from_distance(distance)
        recency = _recency_score(row.get("created_at", datetime.now(timezone.utc).isoformat()))
        importance = row.get("importance", 0.5)
        combined = _combined_score(sim, recency, importance)

        memory = MemoryRecord.from_lance_row(row)
        candidates.append(SearchResult(
            memory=memory,
            raw_similarity=sim,
            recency_score=recency,
            importance_score=importance,
            combined_score=combined,
        ))

    if not candidates:
        return []

    # MMR re-ranking for diversity
    selected = _mmr_rerank(candidates, query_vector, k)

    # Update access counts for returned results
    for result in selected:
        store.update_access(result.memory.id)

    return selected


def _mmr_rerank(
    candidates: list[SearchResult],
    query_vector: list[float],
    k: int,
) -> list[SearchResult]:
    """Maximal Marginal Relevance re-ranking.
    Balances relevance (combined_score) with diversity."""
    if len(candidates) <= k:
        return sorted(candidates, key=lambda r: r.combined_score, reverse=True)

    lam = config.MMR_LAMBDA
    selected: list[SearchResult] = []
    remaining = list(candidates)

    # Precompute candidate vectors
    cand_vectors = np.array(
        [c.memory.vector for c in remaining], dtype=np.float32
    )

    for _ in range(k):
        best_idx = -1
        best_mmr = -float("inf")

        for i, cand in enumerate(remaining):
            relevance = cand.combined_score

            # Max similarity to already-selected items
            if selected:
                sel_vectors = np.array(
                    [s.memory.vector for s in selected], dtype=np.float32
                )
                sims = sel_vectors @ cand_vectors[i]
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0

            mmr_score = lam * relevance - (1 - lam) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))
        cand_vectors = np.delete(cand_vectors, best_idx, axis=0)

    return selected
