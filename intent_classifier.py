"""
Optional ML intent classifier (MiniLM + logistic or MLP head).
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
_MODEL_TYPE = ""
_LOADED = False

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
    global _LOADED, _ENCODER, _CLASSIFIER, _LABELS, _MODEL_TYPE
    if _LOADED:
        return _CLASSIFIER is not None
    _LOADED = True
    if not is_available():
        return False
    try:
        with open(_MODEL_DIR / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        _LABELS = list(meta.get("labels", []))
        _MODEL_TYPE = str(meta.get("model_type", "logistic"))

        from sentence_transformers import SentenceTransformer

        bundle = joblib.load(_MODEL_DIR / "classifier.joblib")
        if isinstance(bundle, dict) and "clf" in bundle:
            _CLASSIFIER = bundle["clf"]
            enc_name = bundle.get("encoder_model", meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
        else:
            _CLASSIFIER = bundle
            enc_name = meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        _ENCODER = SentenceTransformer(enc_name, device="cpu")
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
    try:
        from query_robust import repair_query

        query = repair_query(query)
    except Exception:
        pass
    if not query.strip() or not _load():
        return None
    assert _ENCODER is not None and _CLASSIFIER is not None
    vec = _ENCODER.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if hasattr(_CLASSIFIER, "predict_proba"):
        proba = _CLASSIFIER.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
    else:
        idx = int(_CLASSIFIER.predict(vec)[0])
        conf = 0.85
    if conf < min_confidence:
        return None
    fine = _LABELS[idx] if idx < len(_LABELS) else str(getattr(_CLASSIFIER, "classes_", [idx])[idx])
    collapsed = _COLLAPSE.get(fine, "discussion")
    return collapsed, conf
