"""
Query robustness — typo-tolerant normalization before routing/retrieval.

This is intentionally lightweight (no extra ML training). The chat model does not
need to learn spelling; the backend repairs and fuzzy-matches intent anchors first.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Whole-word replacements (longer keys first when applied with word boundaries).
WORD_REPAIRS: tuple[tuple[str, str], ...] = (
    ("suggets", "suggest"),
    ("sugget", "suggest"),
    ("suggests", "suggest"),
    ("recomend", "recommend"),
    ("recommed", "recommend"),
    ("movei", "movie"),
    ("moive", "movie"),
    ("moveis", "movies"),
    ("moives", "movies"),
    ("flim", "film"),
    ("flims", "films"),
    ("inceptoin", "inception"),
    ("incepion", "inception"),
    ("intersteller", "interstellar"),
    ("directer", "director"),
    ("directers", "director"),
    ("actr", "actor"),
    ("plaot", "plot"),
    ("plott", "plot"),
    ("synopsys", "synopsis"),
    ("recomendation", "recommendation"),
    ("recomendations", "recommendations"),
    ("whoo", "who"),
    ("whos", "who"),
    ("wat", "what"),
    ("wht", "what"),
)

# Glued phrases → spaced (title / grammar).
_GLUED_REPAIRS: tuple[tuple[str, str], ...] = (
    (r"\bthegirl\b", "the girl"),
    (r"\btheboy\b", "the boy"),
    (r"\bthemovie\b", "the movie"),
    (r"\bthefilm\b", "the film"),
)

# Intent / routing anchors: fuzzy-match tokens to these (edit distance via ratio).
_INTENT_ANCHORS: tuple[str, ...] = (
    "who",
    "what",
    "when",
    "suggest",
    "recommend",
    "director",
    "directed",
    "plot",
    "synopsis",
    "story",
    "happens",
    "actor",
    "cast",
    "starring",
    "protagonist",
    "genre",
    "rating",
    "trailer",
    "watch",
    "movies",
    "films",
)

# Minimum similarity to rewrite a token to an anchor (0–1).
_FUZZY_ANCHOR_RATIO = 0.82


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _fuzzy_anchor_token(word: str) -> str:
    """Map a single token to the closest intent anchor when clearly a typo."""
    clean = re.sub(r"[^a-z0-9]", "", word.lower())
    if len(clean) < 4:
        return word
    if clean in _INTENT_ANCHORS:
        return clean
    best = clean
    best_score = 0.0
    for anchor in _INTENT_ANCHORS:
        score = _similar(clean, anchor)
        if score > best_score:
            best_score = score
            best = anchor
    if best_score >= _FUZZY_ANCHOR_RATIO and best != clean:
        return best
    return word


def repair_query(query: str) -> str:
    """
    Normalize user text for routing and retrieval.

    Order: trim → glued words → dictionary typos → fuzzy intent anchors.
    """
    q = (query or "").strip()
    if not q:
        return q
    q = re.sub(r"[/\\|]+$", "", q).strip()
    q = re.sub(r"^[/\\|]+", "", q).strip()
    q = re.sub(r"\s+", " ", q)

    for pat, repl in _GLUED_REPAIRS:
        q = re.sub(pat, repl, q, flags=re.IGNORECASE)

    for wrong, right in WORD_REPAIRS:
        q = re.sub(rf"\b{re.escape(wrong)}\b", right, q, flags=re.IGNORECASE)

    tokens = q.split()
    if tokens:
        q = " ".join(_fuzzy_anchor_token(t) for t in tokens)

    return q.strip()


def fuzzy_phrase_match(query: str, phrase: str, *, ratio: float = 0.88) -> bool:
    """True if phrase appears in query or is approximated by consecutive tokens."""
    q = repair_query(query).lower()
    phrase = phrase.lower().strip()
    if phrase in q:
        return True
    q_tokens = q.split()
    p_tokens = phrase.split()
    if len(p_tokens) > len(q_tokens):
        return False
    width = len(p_tokens)
    for i in range(len(q_tokens) - width + 1):
        chunk = " ".join(q_tokens[i : i + width])
        if _similar(chunk, phrase) >= ratio:
            return True
    return False
