"""Evaluation: Recall@k, Precision@k, MRR on labeled test set."""

import json
from pathlib import Path

import reader
from models import EvalQuery


def _recall_at_k(retrieved_texts: list[str], relevant_texts: list[str]) -> float:
    """Fraction of relevant items found in retrieved set."""
    if not relevant_texts:
        return 0.0
    retrieved_set = set(t.lower().strip() for t in retrieved_texts)
    found = sum(
        1 for rel in relevant_texts
        if any(rel.lower().strip() in ret for ret in retrieved_set)
    )
    return found / len(relevant_texts)


def _precision_at_k(retrieved_texts: list[str], relevant_texts: list[str]) -> float:
    """Fraction of retrieved items that are relevant."""
    if not retrieved_texts:
        return 0.0
    relevant_lower = [r.lower().strip() for r in relevant_texts]
    hits = sum(
        1 for ret in retrieved_texts
        if any(rel in ret.lower().strip() for rel in relevant_lower)
    )
    return hits / len(retrieved_texts)


def _reciprocal_rank(retrieved_texts: list[str], relevant_texts: list[str]) -> float:
    """1/rank of the first relevant item found."""
    relevant_lower = [r.lower().strip() for r in relevant_texts]
    for i, ret in enumerate(retrieved_texts, 1):
        ret_lower = ret.lower().strip()
        if any(rel in ret_lower for rel in relevant_lower):
            return 1.0 / i
    return 0.0


def evaluate(data_path: str, k: int = 10) -> dict:
    """Run evaluation on a labeled JSON dataset.

    Dataset format: [{"query": "...", "relevant_texts": ["...", ...]}]
    """
    path = Path(data_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    with open(path) as f:
        raw = json.load(f)

    queries = [EvalQuery(**item) for item in raw]

    total_recall = 0.0
    total_precision = 0.0
    total_rr = 0.0

    for q in queries:
        results = reader.search(q.query, k=k)
        retrieved = [r.memory.text for r in results]

        total_recall += _recall_at_k(retrieved, q.relevant_texts)
        total_precision += _precision_at_k(retrieved, q.relevant_texts)
        total_rr += _reciprocal_rank(retrieved, q.relevant_texts)

    n = len(queries) or 1
    return {
        "queries": [q.query for q in queries],
        "recall_at_k": total_recall / n,
        "precision_at_k": total_precision / n,
        "mrr": total_rr / n,
        "k": k,
        "num_queries": n,
    }
