"""
Optional ML intent classifier (MiniLM + logistic regression).
Loaded automatically by 03_rag_pipeline.detect_intent() when intent_model/ exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

_ROOT = Path(__file__).resolve().parent
_MODEL_DIR = _ROOT / "intent_model"
_ENCODER = None
_CLASSIFIER = None
_LABELS: list[str] = []
_LOADED = False

# Fine label → respond() intent
_COLLAPSE = {
    "factual_cast": "factual",
    "factual_director": "factual",
    "factual_plot": "factual",
    "factual_year": "factual",
    "factual_other": "factual",
    "recommend": "recommend",
    "discussion": "discussion",
}


def is_available() -> bool:
    return (_MODEL_DIR / "classifier.joblib").is_file() and (_MODEL_DIR / "meta.json").is_file()


def _load() -> bool:
    global _LOADED, _ENCODER, _CLASSIFIER, _LABELS
    if _LOADED:
        return _CLASSIFIER is not None
    _LOADED = True
    if not is_available():
        return False
    try:
        from sentence_transformers import SentenceTransformer

        with open(_MODEL_DIR / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        model_name = meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        _LABELS = list(meta.get("labels", []))
        _ENCODER = SentenceTransformer(model_name, device="cpu")
        _CLASSIFIER = joblib.load(_MODEL_DIR / "classifier.joblib")
        return True
    except Exception:
        _ENCODER = None
        _CLASSIFIER = None
        return False


def predict(query: str, min_confidence: float = 0.52) -> Optional[tuple[str, float]]:
    """
    Return (collapsed_intent, confidence) or None if model missing / low confidence.
    collapsed_intent is one of: factual | recommend | discussion
    """
    if not query.strip() or not _load():
        return None
    assert _ENCODER is not None and _CLASSIFIER is not None
    vec = _ENCODER.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    proba = _CLASSIFIER.predict_proba(vec)[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    if conf < min_confidence:
        return None
    fine = _LABELS[idx] if idx < len(_LABELS) else str(_CLASSIFIER.classes_[idx])
    collapsed = _COLLAPSE.get(fine, "discussion")
    return collapsed, conf
