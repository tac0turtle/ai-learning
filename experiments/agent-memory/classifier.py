"""Memory relationship classification with pluggable backends (NLI model or LLM).

Two backends:
  - "nli":  DeBERTa cross-encoder, ~15-30ms/pair on CPU, no API calls.
            Install with: uv pip install -e ".[nli]"
  - "llm":  Anthropic Claude via tool_use, ~500-1500ms/pair, requires API key.
            More accurate on temporal updates and implicit contradictions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

import config
import embeddings
import store
from models import ClassificationResult, MemoryRelation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ClassifierBackend(ABC):
    @abstractmethod
    def classify(self, new_text: str, existing_text: str) -> ClassificationResult:
        """Classify the relationship between a new memory and an existing one."""


# ---------------------------------------------------------------------------
# NLI backend (DeBERTa cross-encoder)
# ---------------------------------------------------------------------------

_nli_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is not None:
        return _nli_model

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError(
            "NLI backend requires transformers + optimum. "
            "Install with: uv pip install -e '.[nli]'"
        )

    logger.info("Loading NLI model: %s", config.NLI_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
    model.train(False)  # Set to inference mode
    _nli_model = {"tokenizer": tokenizer, "model": model}
    return _nli_model


class NLIClassifier(ClassifierBackend):
    """Cross-encoder NLI model. Maps 3-class output to MemoryRelation.

    DeBERTa label order: entailment=0, neutral=1, contradiction=2
    (check model.config.id2label if using a different checkpoint).
    """

    # DeBERTa-mnli label mapping (verify per checkpoint)
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2

    def classify(self, new_text: str, existing_text: str) -> ClassificationResult:
        import torch

        nli = _get_nli_model()
        tokenizer = nli["tokenizer"]
        model = nli["model"]

        inputs = tokenizer(
            existing_text,  # premise
            new_text,       # hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        p_entail = probs[self.ENTAILMENT].item()
        p_neutral = probs[self.NEUTRAL].item()
        p_contra = probs[self.CONTRADICTION].item()

        # Map NLI output to MemoryRelation
        if p_entail > config.NLI_ENTAILMENT_THRESHOLD:
            return ClassificationResult(
                relation=MemoryRelation.DUPLICATE,
                confidence=p_entail,
                reasoning=f"NLI entailment={p_entail:.3f}",
            )

        if p_contra > config.NLI_CONTRADICTION_THRESHOLD:
            # NLI can't distinguish contradiction from temporal update.
            # We label it "updates" as the safer default -- the new info
            # is more recent and likely supersedes the old.
            return ClassificationResult(
                relation=MemoryRelation.UPDATES,
                confidence=p_contra,
                reasoning=f"NLI contradiction={p_contra:.3f} (treated as update)",
            )

        # Neutral with moderate entailment suggests extension
        if p_entail > 0.3 and p_neutral > 0.3:
            return ClassificationResult(
                relation=MemoryRelation.EXTENDS,
                confidence=max(p_entail, p_neutral),
                reasoning=f"NLI entail={p_entail:.3f} neutral={p_neutral:.3f}",
            )

        return ClassificationResult(
            relation=MemoryRelation.NEW,
            confidence=p_neutral,
            reasoning=f"NLI neutral={p_neutral:.3f}",
        )


# ---------------------------------------------------------------------------
# LLM backend (Anthropic Claude)
# ---------------------------------------------------------------------------

class LLMClassifier(ClassifierBackend):
    """Uses Claude tool_use for nuanced relationship classification.

    Slower (~500-1500ms) but handles temporal updates, implicit
    contradictions, and domain-specific reasoning correctly.
    """

    def classify(self, new_text: str, existing_text: str) -> ClassificationResult:
        from anthropic import Anthropic
        from prompts import CLASSIFY_SYSTEM, CLASSIFY_TOOL

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=256,
            system=CLASSIFY_SYSTEM,
            tools=[CLASSIFY_TOOL],
            messages=[{
                "role": "user",
                "content": (
                    f"EXISTING memory: {existing_text}\n\n"
                    f"NEW memory: {new_text}"
                ),
            }],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "classify_relationship":
                inp = block.input
                try:
                    relation = MemoryRelation(inp["relation"])
                except ValueError:
                    relation = MemoryRelation.NEW
                return ClassificationResult(
                    relation=relation,
                    confidence=inp.get("confidence", 0.5),
                    reasoning=inp.get("reasoning", ""),
                )

        return ClassificationResult(
            relation=MemoryRelation.NEW,
            confidence=0.0,
            reasoning="LLM did not return classification",
        )


# ---------------------------------------------------------------------------
# Factory + pipeline
# ---------------------------------------------------------------------------

_backend: ClassifierBackend | None = None


def _get_backend() -> ClassifierBackend:
    global _backend
    if _backend is not None:
        return _backend

    if config.CLASSIFIER_BACKEND == "nli":
        _backend = NLIClassifier()
    else:
        _backend = LLMClassifier()
    return _backend


def set_backend(backend: str):
    """Override the classifier backend at runtime. Resets cached instance."""
    global _backend
    config.CLASSIFIER_BACKEND = backend
    _backend = None


def find_candidates(new_vector: list[float], limit: int = 5) -> list[dict]:
    """Find existing memories similar enough to warrant classification."""
    existing = store.get_all()
    if not existing:
        return []

    new_vec = np.array(new_vector, dtype=np.float32)
    candidates = []
    for row in existing:
        if not row.get("is_latest", True):
            continue
        if row.get("consolidated_into", ""):
            continue
        ex_vec = np.array(row["vector"], dtype=np.float32)
        sim = float(np.dot(new_vec, ex_vec))
        if sim > config.CLASSIFY_SIMILARITY_THRESHOLD:
            candidates.append((sim, row))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in candidates[:limit]]


def classify_against_existing(
    new_text: str,
    new_vector: list[float],
) -> ClassificationResult:
    """Classify a new memory against existing memories.

    Returns the strongest non-NEW classification found, or NEW if no
    existing memory is related.
    """
    candidates = find_candidates(new_vector)
    if not candidates:
        return ClassificationResult(relation=MemoryRelation.NEW, confidence=1.0)

    backend = _get_backend()
    best: ClassificationResult | None = None

    for row in candidates:
        result = backend.classify(new_text, row["text"])
        result.existing_memory_id = row["id"]

        if result.relation == MemoryRelation.DUPLICATE:
            return result  # Immediate -- skip this memory

        if result.relation != MemoryRelation.NEW:
            if best is None or result.confidence > best.confidence:
                best = result

    return best or ClassificationResult(relation=MemoryRelation.NEW, confidence=1.0)
