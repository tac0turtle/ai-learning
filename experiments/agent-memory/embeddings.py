"""Lazy-loaded sentence-transformer for encoding text to vectors."""

from typing import Optional

import numpy as np

import config

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode a batch of texts into vectors. Returns shape (n, dim)."""
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def encode_single(text: str) -> list[float]:
    """Encode a single text into a vector. Returns a plain list for LanceDB."""
    vec = encode([text])[0]
    return vec.tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))
