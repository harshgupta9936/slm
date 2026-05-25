"""
STEP 3 — RAG: Chroma vector store + grounded chat (llama.cpp GGUF)

Usage:
  python 03_rag_pipeline.py --build --movies databse.csv
  python 03_rag_pipeline.py --chat --model path/to/model.gguf
  python 03_rag_pipeline.py --recommend "slow melancholy sci-fi" --movies databse.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Optional, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

from cinephile_voice import (
    RAG_USER_BODY,
    SYSTEM_PROMPT,
    format_abstain,
    format_cast_lead,
    format_character_portrayal,
    format_director,
    format_director_opinion,
    format_director_web,
    format_disambiguation,
    format_genre,
    format_no_plot,
    format_plot,
    format_protagonist,
    format_rating,
    format_year,
    multi_plot_intro,
    recommend_bullet_why,
    recommend_heading_director,
    recommend_heading_general,
    recommend_no_director,
    recommend_no_matches,
)
from movie_data import load_movies_dataframe
from query_robust import repair_query

CHROMA_DIR = "./movie_vector_store"
COLLECTION = "movies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
CONTEXT_WINDOW = 4096

def format_phi3_prompt(system: str, user: str) -> str:
    """Phi-3 / Phi-3.5 style markers (works with common GGUF chat templates)."""
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n"
    )


def format_raw_prompt(system: str, user: str) -> str:
    return f"{system}\n\n{user}\n\nMr. Cinephile:"


def _docs_and_meta_from_df(df):
    docs, metas, ids = [], [], []
    for i, row in df.iterrows():
        doc = (
            f"{row['title']}. Directed by {row['director']}. {row['overview']} "
            f"Genre: {row['genre']}. Year: {row['year']}. Rating: {row['rating']}."
        )
        docs.append(doc)
        mid = str(row.get("movie_id", i))
        ids.append(f"movie_{mid}")
        metas.append(
            {
                "title": str(row["title"]),
                "director": str(row["director"]),
                "overview": str(row["overview"])[:800],
                "genre": str(row["genre"]),
                "year": str(row["year"]),
                "rating": str(row["rating"]),
                "movie_id": mid,
            }
        )
    return docs, metas, ids


class _NumpyMovieStore:
    """Cosine retrieval with sentence-transformers only (no Chroma)."""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.np_dir = self.persist_dir / "np_rag"
        self._emb_path = self.np_dir / "embeddings.npy"
        self._meta_path = self.np_dir / "meta.json"
        self._model: Optional[SentenceTransformer] = None
        self._embeddings: Optional[np.ndarray] = None
        self._meta: Optional[list] = None

    def _encoder(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
        return self._model

    def build(self, movies_csv: str, batch_size: int = 64):
        df = load_movies_dataframe(movies_csv)
        docs, metas, _ids = _docs_and_meta_from_df(df)
        self.np_dir.mkdir(parents=True, exist_ok=True)
        enc = self._encoder()
        vecs = enc.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(self._emb_path, vecs.astype(np.float32))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)
        self._embeddings = vecs
        self._meta = metas
        print(f"\nNumPy vector store: {len(metas)} movies -> {self.np_dir}")

    def _load(self):
        if self._embeddings is not None and self._meta is not None:
            return
        if not self._emb_path.is_file() or not self._meta_path.is_file():
            raise RuntimeError(
                f"No NumPy index in {self.np_dir}. Run: python 03_rag_pipeline.py --build --movies <csv>"
            )
        self._embeddings = np.load(self._emb_path)
        with open(self._meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        self._load()
        enc = self._encoder()
        q = enc.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        sims = self._embeddings @ q
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for j in idx:
            out.append({**self._meta[int(j)], "relevance_score": round(float(sims[j]), 4)})
        return out

    def list_director_names(self) -> list[str]:
        self._load()
        names: set[str] = set()
        for meta in self._meta:
            for part in _iter_director_parts(str(meta.get("director", ""))):
                names.add(part)
        return sorted(names)

    def movies_by_director(self, director_name: str, limit: int = 20) -> list[dict]:
        self._load()
        wanted = _director_norm(director_name)
        matches: list[dict] = []
        for meta in self._meta:
            if _movie_has_director(str(meta.get("director", "")), wanted):
                matches.append({**meta, "relevance_score": 1.0})
        matches.sort(key=lambda m: _parse_rating_value(m.get("rating", "")), reverse=True)
        return matches[:limit]


class _ChromaMovieStore:
    def __init__(self, persist_dir: str):
        if not _HAS_CHROMA:
            raise RuntimeError("chromadb is not installed")
        self.persist_dir = persist_dir
        self._all_metas: Optional[list[dict]] = None
        self.embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _cached_metadatas(self) -> list[dict]:
        """Load catalogue metadatas once per process (avoid 90s+ on every director query)."""
        if self._all_metas is not None:
            return self._all_metas
        collection = self.client.get_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
        )
        data = collection.get(include=["metadatas"])
        self._all_metas = [m for m in data.get("metadatas", []) if m]
        return self._all_metas

    def build(self, movies_csv: str, batch_size: int = 128):
        df = load_movies_dataframe(movies_csv)
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        collection = self.client.create_collection(
            name=COLLECTION,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        docs, metas, ids = _docs_and_meta_from_df(df)
        total = len(docs)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metas[start:end],
            )
            print(f"  Indexed {end}/{total} movies...", end="\r")
        print(f"\nChroma vector store: {total} movies -> {self.persist_dir}")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        try:
            collection = self.client.get_collection(
                name=COLLECTION,
                embedding_function=self.embed_fn,
            )
        except Exception as e:
            raise RuntimeError(
                f"No Chroma store at '{self.persist_dir}'. Run --build first."
            ) from e
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        out = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            out.append({**meta, "relevance_score": round(1.0 - float(dist), 4)})
        return out

    def list_director_names(self) -> list[str]:
        try:
            names: set[str] = set()
            for meta in self._cached_metadatas():
                for part in _iter_director_parts(str(meta.get("director", ""))):
                    names.add(part)
            return sorted(names)
        except Exception as e:
            raise RuntimeError(
                f"No Chroma store at '{self.persist_dir}'. Run --build first."
            ) from e

    def movies_by_director(self, director_name: str, limit: int = 20) -> list[dict]:
        try:
            wanted = _director_norm(director_name)
            matches: list[dict] = []
            for meta in self._cached_metadatas():
                if _movie_has_director(str(meta.get("director", "")), wanted):
                    matches.append({**meta, "relevance_score": 1.0})
            matches.sort(key=lambda m: _parse_rating_value(m.get("rating", "")), reverse=True)
            return matches[:limit]
        except Exception as e:
            raise RuntimeError(
                f"No Chroma store at '{self.persist_dir}'. Run --build first."
            ) from e


class MovieVectorStore:
    """auto: Chroma if installed, else NumPy + MiniLM."""

    def __init__(self, persist_dir: str = CHROMA_DIR, backend: str = "auto"):
        self._director_names_cache: Optional[list[str]] = None
        if backend == "auto":
            backend = "chroma" if _HAS_CHROMA else "numpy"
        if backend == "chroma":
            self._inner: _ChromaMovieStore | _NumpyMovieStore = _ChromaMovieStore(persist_dir)
            self.backend = "chroma"
        elif backend == "numpy":
            self._inner = _NumpyMovieStore(persist_dir)
            self.backend = "numpy"
        else:
            raise ValueError("backend must be auto|chroma|numpy")

    def build(self, movies_csv: str, batch_size: int = 128):
        self._inner.build(movies_csv, batch_size=batch_size)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        return self._inner.search(query, top_k=top_k)

    def list_director_names(self) -> list[str]:
        if self._director_names_cache is None:
            self._director_names_cache = self._inner.list_director_names()
        return self._director_names_cache

    def movies_by_director(self, director_name: str, limit: int = 20) -> list[dict]:
        return self._inner.movies_by_director(director_name, limit=limit)

    @staticmethod
    def format_context(movies: list[dict]) -> str:
        lines = []
        for m in movies:
            lines.append(
                f"• {m['title']} ({m.get('year', '?')}) — {m.get('director', '?')} "
                f"[{m.get('genre', '?')}]\n  {m.get('overview', '')[:400]}"
            )
        return "\n\n".join(lines)


def is_cast_or_actor_question(query: str) -> bool:
    """True when the user asks about actors, cast, or who plays a role — not directors."""
    ql = query.lower()
    if any(
        sig in ql
        for sig in (
            "lead role",
            "leading role",
            "main role",
            "lead actor",
            "main actor",
            "lead actress",
            "who plays",
            "who played",
            "who portrays",
            "who portrayed",
            "who stars",
            "who starred",
            "cast of",
            "starring",
            "star of",
            "actor in",
            "actress in",
            "who has the lead",
            "who had the lead",
            "played by",
            "main character",
            "protagonist",
        )
    ):
        return True
    if re.search(
        r"\bwho\s+(?:has|had|plays|played|is|was|portrays|portrayed|stars|starred)\b",
        ql,
    ) and re.search(r"\b(actor|role|cast|character|lead|star|stars)\b", ql):
        return True
    return False


def detect_intent_rules(query: str) -> str:
    q = query.lower()
    q_simple = re.sub(r"[^a-z0-9 ]+", " ", q)
    if re.search(
        r"\b(what are|which are|what)\b.*\b(best|top|greatest|highest)\b.*\b(movies|films)\b",
        q,
    ):
        return "recommend"
    if re.search(r"\b(best|top|greatest|highest rated)\b.*\b(movies|films)\b", q):
        return "recommend"
    if re.search(r"\b(movies|films)\b.*\b(by|from|of)\b", q):
        return "recommend"
    if re.search(r"\bsugg?ests?\b", q):
        return "recommend"
    for sig in (
        "recommend",
        "suggest",
        "sugget",
        "recomend",
        "recommed",
        "what should i watch",
        "something like",
        "similar to",
        "good movie",
        "what to watch",
        "movies like",
        "find me",
        "wanna watch",
    ):
        if sig in q:
            return "recommend"
    # Genre-driven asks are usually recommendations even without perfect wording.
    if ("movie" in q_simple or "movies" in q_simple) and any(
        g in q_simple
        for g in (
            "thriller",
            "suspense",
            "horror",
            "comedy",
            "romance",
            "action",
            "drama",
            "sci fi",
            "science fiction",
            "adventure",
            "crime",
            "mystery",
            "fantasy",
        )
    ):
        return "recommend"
    for sig in (
        "who directed",
        "who is the director",
        "director of",
        "who made",
        "lead actor",
        "actor in",
        "actor of",
        "actor",
        "main actor",
        "main lead actor",
        "protagonist",
        "main character",
        "hero of",
        "what year",
        "when was",
        "cast of",
        "starring",
        "plot of",
        "what happens",
        "happens in",
        "what went on",
        "what's it about",
        "whats it about",
        "synopsis",
        "story of",
        "tell me about",
        "who was",
        "who is",
        "who played",
        "who portrays",
        "who portrayed",
        "played by",
    ):
        if sig in q:
            return "factual"
    if re.search(
        r"\b(what happens|happens in|what went on|synopsis|story of|plot of)\b",
        q,
    ):
        return "factual"
    if re.search(r"\b(his|her|their)\b", q) and re.search(
        r"\b(more|movies|films|suggest|recommend|sugget|recomend)\b", q
    ):
        return "recommend"
    return "discussion"


def detect_intent(query: str) -> str:
    """Rules first; ML classifier only when rules are ambiguous."""
    if is_cast_or_actor_question(query):
        return "factual"
    if _is_plot_or_about_question_text(query):
        return "factual"
    ruled = detect_intent_rules(query)
    if ruled in ("factual", "recommend"):
        return ruled
    try:
        import intent_classifier as ic

        hit = ic.predict(query)
        if hit is not None:
            intent, conf = hit
            if conf >= 0.58:
                return intent
    except Exception:
        pass
    return ruled


# Skip slow ML intent classifier when rules already know the route.
_PLOT_OR_ABOUT_SIGS = (
    "what happens",
    "happens in",
    "what went on",
    "plot of",
    "story of",
    "synopsis",
    "tell me about",
    "about the movie",
    "about the film",
)
_OPINION_SIGS = (
    "what do you think",
    "your opinion",
    "thoughts on",
    "how is the movie",
    "how was the movie",
)


def _is_plot_or_about_question_text(user_query: str) -> bool:
    qn = user_query.lower()
    return any(sig in qn for sig in _PLOT_OR_ABOUT_SIGS)


def _is_opinion_question_text(user_query: str) -> bool:
    qn = user_query.lower()
    return any(sig in qn for sig in _OPINION_SIGS)


def normalize_user_query(query: str) -> str:
    """Typo-tolerant cleanup before routing, retrieval, and web lookup (see query_robust.py)."""
    return repair_query(query)


def _extract_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None


def _movie_record_year(movie: dict) -> Optional[int]:
    return _extract_year_from_text(f"{movie.get('year', '')} {movie.get('title', '')}")


def _norm_title_query(text: str) -> str:
    q = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    q = q.replace("spiderman", "spider man")
    q = q.replace("spiderverse", "spider verse")
    q = re.sub(r"\bin to\b", "into", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _director_norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", name.lower())).strip()


def _iter_director_parts(director_field: str):
    for part in re.split(r"\s*\|\s*", director_field):
        name = part.strip()
        if name and name.lower() != "unknown":
            yield name


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _director_norm(a), _director_norm(b)).ratio()


def _director_matches(movie_director: str, wanted: str) -> bool:
    """Require the full director name phrase (e.g. 'peter jackson'), not just 'jackson'."""
    md = _director_norm(movie_director)
    w = _director_norm(wanted)
    if not w:
        return False
    if len(w.split()) < 2:
        return w in md
    return w in md


def _movie_has_director(movie_director: str, wanted: str) -> bool:
    w = _director_norm(wanted)
    for part in _iter_director_parts(movie_director):
        if _director_matches(part, w):
            return True
        if len(w.split()) >= 2 and _name_similarity(part, w) >= 0.82:
            return True
    return _director_matches(movie_director, w)


def resolve_director_movies(
    vs: "MovieVectorStore", raw_name: str, limit: int = 12
) -> tuple[str, list[dict], bool]:
    """Return (canonical director name, movies, typo_was_corrected)."""
    exact = vs.movies_by_director(raw_name, limit=limit)
    if exact:
        return raw_name.strip().title(), exact, False

    raw_n = _director_norm(raw_name)
    best_name: Optional[str] = None
    best_score = 0.0
    for candidate in vs.list_director_names():
        cn = _director_norm(candidate)
        if raw_n in cn or cn in raw_n:
            best_name = candidate
            best_score = 1.0
            break
        score = _name_similarity(candidate, raw_name)
        if score > best_score:
            best_score = score
            best_name = candidate

    if best_name and best_score >= 0.78:
        movies = vs.movies_by_director(best_name, limit=limit)
        corrected = _director_norm(best_name) != raw_n
        return best_name, movies, corrected

    return raw_name.strip().title(), [], False


def _parse_rating_value(rating: object) -> float:
    try:
        return float(str(rating).strip())
    except (TypeError, ValueError):
        return 0.0


def _same_film_title(a: str, b: str) -> bool:
    """True when two strings refer to the same film title (fuzzy)."""
    an = _norm_title_query(a)
    bn = _norm_title_query(b)
    if not an or not bn:
        return False
    if an == bn or an in bn or bn in an:
        return True
    at = {w for w in an.split() if len(w) > 2}
    bt = {w for w in bn.split() if len(w) > 2}
    if not at or not bt:
        return False
    return len(at & bt) / min(len(at), len(bt)) >= 0.75


def _title_match_boost(user_query: str, title: str) -> float:
    q = _norm_title_query(user_query)
    t = _norm_title_query(title)
    boost = 0.0
    if "into" in q and "into" in t:
        boost += 7.0
    if "across" in q and "across" in t:
        boost += 7.0
    if "into" in q and "across" in t and "across" not in q:
        boost -= 6.0
    if "across" in q and "into" in t and "into" not in q:
        boost -= 6.0
    if "spider" in q and "verse" in q and "spider" in t and "verse" in t:
        boost += 8.0
    q_tokens = {w for w in q.split() if len(w) > 2}
    t_tokens = {w for w in t.split() if len(w) > 2}
    boost += 2.5 * len(q_tokens & t_tokens)
    return boost


def identify_primary_film(user_query: str, retrieved: list[dict]) -> Optional[dict]:
    """Pick the single film the user is most likely discussing."""
    if not retrieved:
        return None
    q_norm = _norm_title_query(user_query)
    q_tokens = {t for t in q_norm.split() if len(t) > 2}
    asked_year = None
    ym = re.search(r"\b(19\d{2}|20\d{2})\b", user_query)
    if ym:
        asked_year = int(ym.group(1))

    def _score(m: dict) -> tuple[float, int]:
        title = str(m.get("title", ""))
        title_tokens = set(_norm_title_query(title).split())
        overlap = len(q_tokens & title_tokens)
        score = overlap * 10 + float(m.get("relevance_score", 0.0)) + _title_match_boost(user_query, title)
        m_year_s = str(m.get("year", ""))
        ym2 = re.search(r"\b(19\d{2}|20\d{2})\b", m_year_s + " " + title)
        if asked_year is not None and ym2:
            m_year = int(ym2.group(1))
            if m_year == asked_year:
                score += 12.0
            else:
                score -= 18.0
        return score, overlap

    if asked_year is not None:
        year_hits: list[tuple[float, dict]] = []
        for m in retrieved:
            sc, overlap = _score(m)
            m_year = _movie_record_year(m)
            if m_year == asked_year and overlap >= 1:
                year_hits.append((sc, m))
        if year_hits:
            year_hits.sort(key=lambda x: x[0], reverse=True)
            return year_hits[0][1]
        return None

    best = retrieved[0]
    best_score = -1.0
    for m in retrieved:
        sc, _ = _score(m)
        if sc > best_score:
            best = m
            best_score = sc
    return best


def _web_hit_to_movie(web: dict, *, relevance_score: float = 0.96) -> dict:
    """Normalize TMDB/Wikipedia hit into the same shape as vector-store rows."""
    mid = str(web.get("movie_id", "")).strip()
    return {
        "title": str(web.get("title", "")),
        "year": str(web.get("year", "")),
        "director": str(web.get("director", "")),
        "overview": str(web.get("overview", "")),
        "genre": "",
        "rating": "",
        "relevance_score": relevance_score,
        "movie_id": mid,
        "catalog_source": str(web.get("source", "web")),
    }


def _http_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CineBot/1.0 (+local-rag-app)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_web_movie_overview(
    title: str,
    *,
    year: Optional[int] = None,
) -> Optional[dict]:
    """
    Plot / metadata from the open web when the local CSV has no match.
    Order: TMDB (if TMDB_API_KEY) → Wikipedia → Wikidata.
    Returns {title, year, director, overview, source}.
    """
    title = re.sub(r"\s+", " ", title.strip())
    if len(title) < 2:
        return None

    def _pack(
        *,
        t: str,
        y: str = "",
        director: str = "",
        overview: str = "",
        source: str,
        movie_id: str = "",
    ) -> Optional[dict]:
        ov = overview.strip()
        if len(ov) < 40:
            return None
        out = {
            "title": t.strip() or title,
            "year": y.strip(),
            "director": director.strip(),
            "overview": ov[:1200],
            "source": source,
        }
        if movie_id:
            out["movie_id"] = movie_id
        return out

    tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()
    if tmdb_key:
        try:
            search_params: dict[str, str] = {"api_key": tmdb_key, "query": title}
            if year is not None:
                search_params["year"] = str(year)
            params = urllib.parse.urlencode(search_params)
            data = _http_json(
                f"https://api.themoviedb.org/3/search/movie?{params}",
                timeout=6.0,
            )
            for row in data.get("results", [])[:6]:
                rel = str(row.get("release_date", ""))[:4]
                if year is not None and rel.isdigit() and int(rel) != year:
                    continue
                mid = row.get("id")
                if not mid:
                    continue
                detail = _http_json(
                    f"https://api.themoviedb.org/3/movie/{mid}?"
                    + urllib.parse.urlencode({"api_key": tmdb_key}),
                    timeout=6.0,
                )
                ov = str(detail.get("overview", "")).strip()
                directors: list[str] = []
                try:
                    credits = _http_json(
                        f"https://api.themoviedb.org/3/movie/{mid}/credits?"
                        + urllib.parse.urlencode({"api_key": tmdb_key}),
                        timeout=5.0,
                    )
                    for person in credits.get("crew", []):
                        if str(person.get("job", "")).lower() == "director":
                            name = str(person.get("name", "")).strip()
                            if name:
                                directors.append(name)
                            if len(directors) >= 2:
                                break
                except Exception:
                    pass
                hit = _pack(
                    t=str(detail.get("title", title)),
                    y=rel or str(year or ""),
                    director=", ".join(directors),
                    overview=ov,
                    source="tmdb",
                    movie_id=str(mid),
                )
                if hit:
                    return hit
        except Exception:
            pass

    try:
        sr = f"{title} {year} film" if year is not None else f"{title} film"
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": sr,
                "format": "json",
                "srlimit": 6,
                "utf8": 1,
            }
        )
        search_data = _http_json(f"https://en.wikipedia.org/w/api.php?{params}", timeout=7.0)
        hits = search_data.get("query", {}).get("search", [])
        for hit in hits:
            page_title = str(hit.get("title", "")).strip()
            snippet = str(hit.get("snippet", "")).lower()
            if year is not None and str(year) not in snippet and str(year) not in page_title:
                continue
            ext_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "exintro": True,
                    "titles": page_title,
                    "format": "json",
                }
            )
            ext_data = _http_json(
                f"https://en.wikipedia.org/w/api.php?{ext_params}",
                timeout=7.0,
            )
            pages = ext_data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = str(page.get("extract", "")).strip()
                if len(extract) < 40:
                    continue
                y = str(year or "")
                ym = re.search(r"\b(19\d{2}|20\d{2})\b", extract[:200])
                if ym:
                    y = ym.group(1)
                packed = _pack(
                    t=page_title.replace(" (film)", "").strip(),
                    y=y,
                    overview=extract,
                    source="wikipedia",
                )
                if packed:
                    return packed
    except Exception:
        pass

    try:
        params = urllib.parse.urlencode(
            {
                "action": "wbsearchentities",
                "search": f"{title} {year}".strip() if year else title,
                "language": "en",
                "type": "item",
                "format": "json",
                "limit": 5,
            }
        )
        search_data = _http_json(f"https://www.wikidata.org/w/api.php?{params}", timeout=7.0)
        for it in search_data.get("search", []):
            desc = str(it.get("description", "")).lower()
            if "film" not in desc and "movie" not in desc:
                continue
            label = str(it.get("label", title))
            if year is not None:
                y_hit = _extract_year_from_text(desc) or _extract_year_from_text(label)
                if y_hit is not None and y_hit != year:
                    continue
            qid = it.get("id")
            if not qid:
                continue
            ent_params = urllib.parse.urlencode(
                {
                    "action": "wbgetentities",
                    "ids": qid,
                    "languages": "en",
                    "format": "json",
                    "props": "claims|descriptions",
                }
            )
            ent = _http_json(f"https://www.wikidata.org/w/api.php?{ent_params}", timeout=7.0)
            entity = ent.get("entities", {}).get(qid, {})
            overview = str(entity.get("descriptions", {}).get("en", {}).get("value", "")).strip()
            rel_year = ""
            for c in entity.get("claims", {}).get("P577", []):
                t = str(c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", ""))
                y = _extract_year_from_text(t)
                if y is not None:
                    rel_year = str(y)
                    break
            directors: list[str] = []
            for c in entity.get("claims", {}).get("P57", [])[:2]:
                v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(v, dict) and "id" in v:
                    lp = urllib.parse.urlencode(
                        {
                            "action": "wbgetentities",
                            "ids": v["id"],
                            "languages": "en",
                            "format": "json",
                            "props": "labels",
                        }
                    )
                    dd = _http_json(f"https://www.wikidata.org/w/api.php?{lp}", timeout=5.0)
                    name = (
                        dd.get("entities", {})
                        .get(v["id"], {})
                        .get("labels", {})
                        .get("en", {})
                        .get("value")
                    )
                    if name:
                        directors.append(name)
            packed = _pack(
                t=label,
                y=rel_year or (str(year) if year else ""),
                director=", ".join(directors),
                overview=overview,
                source="wikidata",
            )
            if packed:
                return packed
    except Exception:
        pass

    return None


def fetch_web_movie_candidates(
    title: str,
    *,
    year: Optional[int] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Return up to `limit` distinct film matches (year + overview) for disambiguation.
    Uses TMDB when TMDB_API_KEY is set, then Wikipedia film pages.
    """
    title = re.sub(r"\s+", " ", title.strip())
    if len(title) < 2:
        return []

    def _pack(
        *,
        t: str,
        y: str = "",
        director: str = "",
        overview: str = "",
        source: str,
    ) -> Optional[dict]:
        ov = overview.strip()
        if len(ov) < 40:
            return None
        return {
            "title": t.strip() or title,
            "year": y.strip(),
            "director": director.strip(),
            "overview": ov[:1200],
            "source": source,
        }

    out: list[dict] = []
    seen_years: set[int] = set()

    def _add(hit: Optional[dict]) -> None:
        if not hit or len(out) >= limit:
            return
        if not _same_film_title(title, str(hit.get("title", ""))):
            return
        y = _extract_year_from_text(str(hit.get("year", "")))
        if y is None or y in seen_years:
            return
        if year is not None and y != year:
            return
        seen_years.add(y)
        out.append(hit)

    tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()
    if tmdb_key:
        try:
            search_params: dict[str, str] = {"api_key": tmdb_key, "query": title}
            if year is not None:
                search_params["year"] = str(year)
            params = urllib.parse.urlencode(search_params)
            data = _http_json(
                f"https://api.themoviedb.org/3/search/movie?{params}",
                timeout=6.0,
            )
            for row in data.get("results", [])[:8]:
                rel = str(row.get("release_date", ""))[:4]
                mid = row.get("id")
                if not mid:
                    continue
                detail = _http_json(
                    f"https://api.themoviedb.org/3/movie/{mid}?"
                    + urllib.parse.urlencode({"api_key": tmdb_key}),
                    timeout=6.0,
                )
                ov = str(detail.get("overview", "")).strip()
                directors: list[str] = []
                try:
                    credits = _http_json(
                        f"https://api.themoviedb.org/3/movie/{mid}/credits?"
                        + urllib.parse.urlencode({"api_key": tmdb_key}),
                        timeout=5.0,
                    )
                    for person in credits.get("crew", []):
                        if str(person.get("job", "")).lower() == "director":
                            name = str(person.get("name", "")).strip()
                            if name:
                                directors.append(name)
                            if len(directors) >= 2:
                                break
                except Exception:
                    pass
                _add(
                    _pack(
                        t=str(detail.get("title", title)),
                        y=rel,
                        director=", ".join(directors),
                        overview=ov,
                        source="tmdb",
                    )
                )
        except Exception:
            pass

    try:
        sr = f"{title} {year} film" if year is not None else f"{title} film"
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": sr,
                "format": "json",
                "srlimit": 8,
                "utf8": 1,
            }
        )
        search_data = _http_json(f"https://en.wikipedia.org/w/api.php?{params}", timeout=7.0)
        for hit in search_data.get("query", {}).get("search", []):
            if len(out) >= limit:
                break
            page_title = str(hit.get("title", "")).strip()
            if "film" not in page_title.lower() and "film" not in str(hit.get("snippet", "")).lower():
                continue
            ext_params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": True,
                    "exintro": True,
                    "titles": page_title,
                    "format": "json",
                }
            )
            ext_data = _http_json(
                f"https://en.wikipedia.org/w/api.php?{ext_params}",
                timeout=7.0,
            )
            pages = ext_data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = str(page.get("extract", "")).strip()
                if len(extract) < 40:
                    continue
                y = ""
                ym = re.search(r"\((\d{4})\s+film\)", page_title, flags=re.IGNORECASE)
                if ym:
                    y = ym.group(1)
                if not y:
                    ym2 = re.search(r"\b(19\d{2}|20\d{2})\b", extract[:200])
                    if ym2:
                        y = ym2.group(1)
                clean_title = re.sub(r"\s*\(\d{4}\s+film\)\s*$", "", page_title, flags=re.IGNORECASE)
                clean_title = clean_title.replace(" (film)", "").strip()
                _add(
                    _pack(
                        t=clean_title,
                        y=y,
                        overview=extract,
                        source="wikipedia",
                    )
                )
    except Exception:
        pass

    out.sort(key=lambda h: _extract_year_from_text(str(h.get("year", ""))) or 0)
    return out


def fetch_trailer_youtube(
    *,
    title: str,
    year: str = "",
    movie_id: str = "",
) -> dict:
    """
    Resolve a YouTube trailer for embedding.
    Returns {youtube_id, embed_url, search_url, source}.
    """
    title = title.strip()
    year = str(year or "").strip()
    movie_id = str(movie_id or "").strip()
    ym = re.search(r"\((19\d{2}|20\d{2})\)\s*$", title)
    if ym and not year:
        year = ym.group(1)
        title = title[: ym.start()].strip()
    search_q = urllib.parse.quote_plus(f"{title} {year} official trailer".strip())

    def _tmdb_trailer_from_id(tmdb_id: str) -> Optional[dict]:
        tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()
        if not tmdb_key or not tmdb_id.isdigit():
            return None
        try:
            params = urllib.parse.urlencode({"api_key": tmdb_key})
            data = _http_json(f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?{params}")
            for v in data.get("results", []):
                if str(v.get("site", "")).lower() == "youtube" and str(v.get("type", "")).lower() in (
                    "trailer",
                    "teaser",
                ):
                    vid = str(v.get("key", "")).strip()
                    if vid:
                        return {
                            "youtube_id": vid,
                            "embed_url": f"https://www.youtube.com/embed/{vid}?autoplay=1&rel=0",
                            "search_url": f"https://www.youtube.com/results?search_query={search_q}",
                            "source": "tmdb",
                        }
        except Exception:
            return None
        return None

    if movie_id.isdigit():
        hit = _tmdb_trailer_from_id(movie_id)
        if hit:
            return hit

    tmdb_key = os.environ.get("TMDB_API_KEY", "").strip()
    if tmdb_key and title:
        try:
            search_params = {"api_key": tmdb_key, "query": title}
            if year.isdigit():
                search_params["year"] = year
            params = urllib.parse.urlencode(search_params)
            data = _http_json(f"https://api.themoviedb.org/3/search/movie?{params}")
            for row in data.get("results", [])[:5]:
                hit = _tmdb_trailer_from_id(str(row.get("id", "")))
                if hit:
                    return hit
        except Exception:
            pass

    return {
        "youtube_id": "",
        "embed_url": "",
        "search_url": f"https://www.youtube.com/results?search_query={search_q}",
        "source": "search",
    }


class NerdGenerator(Protocol):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.65) -> str: ...


class MovieNerdModel:
    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "Install llama-cpp-python (optional: CMAKE_ARGS=-DLLAMA_CUDA=on for GPU)"
            ) from e

        print(f"Loading GGUF: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_WINDOW,
            n_gpu_layers=n_gpu_layers,
            n_threads=8,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.65) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|user|>", "<|end|>", "User:", "\nUser:"],
        )
        return out["choices"][0]["text"].strip()


class HFMovieNerdModel:
    """
    Hugging Face causal LM inference for a merged model folder (recommended when GGUF output is garbled).

    Expects a local directory containing config + tokenizer + weights (e.g. movie-nerd-lora/merged-model).
    """

    def __init__(self, model_dir: str, load_in_4bit: bool = True, attn_implementation: str = "eager"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"HF model dir not found: {path}")

        print(f"Loading HF model: {path}")
        try:
            # Prefer built-in tokenizer implementation first; custom remote tokenizer
            # code can decode merged local models incorrectly in some environments.
            tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
        except Exception:
            tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def _load_model(trust_remote_code: bool):
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.float16
                quant = None
                if load_in_4bit:
                    quant = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                return AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=dtype,
                    quantization_config=quant,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                )
            return AutoModelForCausalLM.from_pretrained(
                str(path),
                torch_dtype="auto",
                device_map={"": "cpu"},
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

        try:
            model = _load_model(trust_remote_code=True)
        except OSError as e:
            msg = str(e)
            if "modeling_phi3.py" not in msg:
                raise
            print("  HF fallback: missing modeling_phi3.py, retrying with built-in Phi-3 loader.")
            cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=False)
            if hasattr(cfg, "auto_map"):
                # transformers expects a mapping here; None triggers TypeError in some versions
                cfg.auto_map = {}
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.float16
                quant = None
                if load_in_4bit:
                    quant = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    config=cfg,
                    torch_dtype=dtype,
                    quantization_config=quant,
                    device_map=device_map,
                    trust_remote_code=False,
                    attn_implementation=attn_implementation,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    config=cfg,
                    torch_dtype="auto",
                    device_map={"": "cpu"},
                    trust_remote_code=False,
                    attn_implementation=attn_implementation,
                )

        model.eval()
        self.tok = tok
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.65) -> str:
        import torch

        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        in_len = int(inputs["input_ids"].shape[-1])

        do_sample = temperature > 0.01
        gen = self.model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=0.9 if do_sample else None,
            repetition_penalty=1.12,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        new_tokens = gen[0, in_len:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


class MovieNerdChat:
    def __init__(
        self,
        vector_store,
        model: Optional[NerdGenerator],
        prompt_format: Literal["phi3", "raw"] = "phi3",
        *,
        use_generative: bool = False,
    ):
        self.vs = vector_store
        self.model = model
        self.history: list[dict] = []
        self.prompt_format = prompt_format
        self.use_generative = use_generative and model is not None

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()

    def _pick_best_movie(self, user_query: str, retrieved: list[dict]) -> Optional[dict]:
        if not retrieved:
            return None
        q = self._norm(user_query)
        q_tokens = set(t for t in q.split() if len(t) > 2)
        if not q_tokens:
            return retrieved[0]
        best = retrieved[0]
        best_score = -1
        for m in retrieved:
            title_tokens = set(self._norm(str(m.get("title", ""))).split())
            overlap = len(q_tokens & title_tokens)
            score = overlap * 10 + float(m.get("relevance_score", 0.0))
            if score > best_score:
                best = m
                best_score = score
        return best

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _part_hint(text: str) -> Optional[int]:
        q = text.lower()
        if "part one" in q or "part 1" in q:
            return 1
        if "part two" in q or "part 2" in q:
            return 2
        return None

    def _best_match_with_constraints(self, user_query: str, retrieved: list[dict]) -> Optional[dict]:
        if not retrieved:
            return None
        asked_year = self._extract_year(user_query)
        asked_part = self._part_hint(user_query)
        q_tokens = set(t for t in self._norm(user_query).split() if len(t) > 2)

        best = None
        best_score = -1.0
        for m in retrieved:
            title = str(m.get("title", ""))
            t_tokens = set(self._norm(title).split())
            overlap = len(q_tokens & t_tokens)
            score = overlap * 10 + float(m.get("relevance_score", 0.0))

            m_year = self._extract_year(str(m.get("year", "")) + " " + title)
            m_part = self._part_hint(title)
            if asked_year is not None and m_year is not None and asked_year != m_year:
                score -= 100
            if asked_part is not None and m_part is not None and asked_part != m_part:
                score -= 100

            if score > best_score:
                best = m
                best_score = score

        if best is not None:
            b_title = str(best.get("title", ""))
            b_year = self._extract_year(str(best.get("year", "")) + " " + b_title)
            b_part = self._part_hint(b_title)
            if asked_year is not None and b_year is not None and asked_year != b_year:
                return None
            if asked_part is not None and b_part is not None and asked_part != b_part:
                return None
        return best

    @staticmethod
    def _extract_movie_query_text(user_query: str) -> str:
        q = user_query.strip()
        for pat in (
            r"\bwhat do you think of(?: the)?(?: movie|film)?\s+(.+)$",
            r"\b(?:your )?opinion (?:on|of)(?: the)?(?: movie|film)?\s+(.+)$",
            r"\bthoughts on(?: the)?(?: movie|film)?\s+(.+)$",
            r"\bhow (?:is|was)\s+(?:the\s+)?(?:movie|film)\s+(.+)$",
            r"\b(?:what is|what's|whats)\s+the\s+(?:plot|story|synopsis)\s+of\s+(?:the\s+)?(.+)$",
            r"\b(?:plot|story|synopsis|summary)\s+of\s+(?:the\s+)?(.+)$",
            r"\btell me about\s+(?:the\s+)?(?:plot\s+of\s+)?(?:the\s+)?(.+)$",
            r"\bwhat happens in\s+(?:the\s+)?(.+)$",
        ):
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                title = m.group(1).strip(" ?.")
                title = re.sub(
                    r"\b(?:from|in|released in)\s+(19\d{2}|20\d{2})\b\s*$",
                    "",
                    title,
                    flags=re.IGNORECASE,
                ).strip(" ?.,")
                return title
        m = re.search(r"\b(?:of|for|about|in)\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            title = m.group(1).strip(" ?.")
            title = re.sub(
                r"\b(?:from|in|released in)\s+(19\d{2}|20\d{2})\b\s*$",
                "",
                title,
                flags=re.IGNORECASE,
            ).strip(" ?.,")
            return title
        q = re.sub(
            r"^\s*(tell me|please|can you|could you|who is|what is|what are|give me)\s+",
            "",
            user_query,
            flags=re.IGNORECASE,
        )
        title = q.strip(" ?.")
        title = re.sub(
            r"\b(?:from|in|released in)\s+(19\d{2}|20\d{2})\b\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip(" ?.,")
        return title

    @staticmethod
    def _with_source(text: str, source: str) -> str:
        return f"{text}\n\nSource used: {source}"

    def _parse_query_spec(self, user_query: str) -> dict:
        qn = self._norm(user_query)
        raw_title = self._extract_movie_query_text(user_query)
        title_tokens = [t for t in self._norm(raw_title).split() if len(t) > 2]
        stop = {
            "movie",
            "film",
            "the",
            "tell",
            "me",
            "who",
            "what",
            "which",
            "is",
            "was",
            "are",
            "in",
            "of",
            "for",
            "about",
            "plot",
            "plaot",
            "story",
            "synopsis",
            "summary",
            "actor",
            "director",
            "protagonist",
            "main",
            "lead",
            "part",
            "one",
            "two",
            "character",
            "starring",
        }
        title_tokens = [t for t in title_tokens if t not in stop]
        title_tokens = [
            t
            for t in title_tokens
            if t not in {"his", "her", "their", "more", "some", "any", "other", "another"}
        ]
        asks_actor = is_cast_or_actor_question(user_query) or any(
            sig in qn
            for sig in ("actor", "lead actor", "main actor", "cast", "starring")
        )
        char_match = re.search(
            r"\bwho\s+(?:was|is|played|portrays?|portrayed)\s+(.+?)\s+in\s+(.+)$",
            user_query,
            flags=re.IGNORECASE,
        )
        asks_character = char_match is not None
        character_name = char_match.group(1).strip(" ?.") if char_match else ""
        character_film = char_match.group(2).strip(" ?.") if char_match else ""
        asks_protagonist = any(sig in qn for sig in ("protagonist", "main character", "hero of"))
        asks_director = any(sig in qn for sig in ("director", "who directed", "who made"))
        asks_year = any(sig in qn for sig in ("what year", "when was", "release year", "came out"))
        asks_plot = any(
            sig in qn
            for sig in (
                "plot",
                "summary",
                "tell me about",
                "about",
                "story of",
                "what happens",
                "happens in",
                "what went on",
                "synopsis",
            )
        )
        asks_multi_plot = asks_plot and any(
            sig in qn for sig in ("movies", "films", "trilogy", "series", "all three", "each")
        )
        asks_genre = any(sig in qn for sig in ("genre", "what kind of movie"))
        asks_rating = any(sig in qn for sig in ("rating", "score", "how good"))
        return {
            "year": self._extract_year(user_query),
            "part": self._part_hint(user_query),
            "title_tokens": title_tokens,
            "asks_actor": asks_actor,
            "asks_character": asks_character,
            "character_name": character_name,
            "character_film": character_film,
            "asks_protagonist": asks_protagonist,
            "asks_director": asks_director,
            "asks_year": asks_year,
            "asks_plot": asks_plot,
            "asks_multi_plot": asks_multi_plot,
            "asks_genre": asks_genre,
            "asks_rating": asks_rating,
        }

    def _select_factual_candidate(self, user_query: str, retrieved: list[dict]) -> tuple[Optional[dict], bool]:
        if not retrieved:
            return None, False
        spec = self._parse_query_spec(user_query)
        title_tokens = set(spec["title_tokens"])
        asked_year = spec["year"]
        asked_part = spec["part"]

        ranked: list[tuple[float, dict, int]] = []
        for m in retrieved:
            title = str(m.get("title", ""))
            t_tokens = set(self._norm(title).split())
            overlap = len(title_tokens & t_tokens)
            rel = float(m.get("relevance_score", 0.0))
            score = rel + overlap * 0.08 + _title_match_boost(user_query, title) * 0.05

            m_year = self._extract_year(f"{m.get('year', '')} {title}")
            m_part = self._part_hint(title)
            if asked_year is not None and m_year is not None and asked_year != m_year:
                score -= 1.0
            if asked_part is not None and m_part is not None and asked_part != m_part:
                score -= 1.0
            ranked.append((score, m, overlap))

        ranked.sort(key=lambda x: x[0], reverse=True)
        best_score, best_movie, best_overlap = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -999.0
        margin = best_score - second_score

        best_title = str(best_movie.get("title", ""))
        best_year = self._extract_year(f"{best_movie.get('year', '')} {best_title}")
        best_part = self._part_hint(best_title)

        if asked_year is not None and best_year is not None and asked_year != best_year:
            return None, False
        if asked_part is not None and best_part is not None and asked_part != best_part:
            return None, False
        if title_tokens and best_overlap == 0:
            return None, False
        if float(best_movie.get("relevance_score", 0.0)) < 0.42:
            return None, False
        if margin < 0.03 and best_overlap <= 1:
            return None, False
        return best_movie, True

    def _abstain(self, user_query: str, source: str = "dataset+web (unverified)") -> str:
        return self._with_source(format_abstain(user_query), source)

    @staticmethod
    def _is_plot_or_about_question(user_query: str) -> bool:
        return _is_plot_or_about_question_text(user_query)

    def _plot_answer_from_primary(self, primary: dict) -> Optional[str]:
        ov = str(primary.get("overview", "")).strip()
        if not ov:
            return None
        return self._with_source(
            format_plot(
                str(primary.get("title", "Unknown")),
                str(primary.get("year", "?")),
                ov,
                str(primary.get("director", "")),
            ),
            self._catalog_source_label(primary),
        )

    @staticmethod
    def _is_recommend_query(user_query: str) -> bool:
        return detect_intent_rules(user_query) == "recommend"

    def _extract_person_for_opinion(self, user_query: str) -> Optional[str]:
        for pat in (
            r"what do you think of\s+(.+?)\??\s*$",
            r"thoughts on\s+(.+?)\??\s*$",
            r"your opinion (?:on|of)\s+(.+?)\??\s*$",
            r"how (?:is|was)\s+(.+?)\s+as a director",
        ):
            m = re.search(pat, user_query.strip(), flags=re.IGNORECASE)
            if not m:
                continue
            name = re.sub(
                r"\b(as a director|the director|movies|films)\b",
                "",
                m.group(1),
                flags=re.IGNORECASE,
            ).strip(" ?.,")
            if name and self._looks_like_director_name(name):
                return name
        return None

    def _resolve_followup_director(self, user_query: str) -> Optional[str]:
        """'more of his movies' → director from the previous turn."""
        ql = user_query.lower()
        if not re.search(r"\b(his|her|their)\b", ql):
            return None
        if not re.search(r"\b(more|movies|films|suggest|recommend|sugget|recomend)\b", ql):
            return None
        for turn in reversed(self.history[-10:]):
            text = str(turn.get("content", ""))
            m = re.search(
                r"directed by\s+([A-Za-z]+(?:\s+[A-Za-z]+)+)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                name = m.group(1).strip().title()
                if self._looks_like_director_name(name):
                    return name
            m2 = re.search(r"That's '([^']+)'", text)
            if m2:
                hits = self.vs.search(m2.group(1), top_k=1)
                if hits:
                    d = str(hits[0].get("director", "")).split("|")[0].strip()
                    if d and self._looks_like_director_name(d):
                        return d
        return None

    def _format_director_opinion_answer(self, director: str, movies: list[dict]) -> str:
        ranked = sorted(
            movies,
            key=lambda m: _parse_rating_value(m.get("rating", "")),
            reverse=True,
        )
        lines = []
        for i, m in enumerate(ranked[:5], start=1):
            title = str(m.get("title", "Unknown"))
            year = str(m.get("year", "?"))
            rating = str(m.get("rating", "?"))
            ov = str(m.get("overview", "")).strip()
            short = ov[:140].rstrip() + ("..." if len(ov) > 140 else "")
            line = f"{i}. {title} ({year}) · {rating}/10 in your catalogue"
            if short:
                line += recommend_bullet_why(short)
            lines.append(line)
        return self._with_source(format_director_opinion(director, lines), "dataset")

    def _try_fast_recommend(self, user_query: str, retrieved: list[dict]) -> Optional[str]:
        director = self._extract_director_constraint(user_query) or self._resolve_followup_director(
            user_query
        )
        if director:
            canon, movies, corrected = resolve_director_movies(self.vs, director, limit=12)
            if movies:
                heading = recommend_heading_director(canon, corrected)
                ranked = sorted(
                    movies,
                    key=lambda m: _parse_rating_value(m.get("rating", "")),
                    reverse=True,
                )
                lines = []
                for i, m in enumerate(ranked[:5], start=1):
                    title = str(m.get("title", "Unknown title"))
                    year = str(m.get("year", "?"))
                    genre = str(m.get("genre", "Unknown genre"))
                    rating = str(m.get("rating", "?"))
                    ov = str(m.get("overview", "")).strip()
                    short_ov = ov[:180].rstrip() + ("..." if len(ov) > 180 else "")
                    bullet = (
                        f"{i}. {title} ({year}) — {canon} [{genre}] · rating {rating}"
                        + (recommend_bullet_why(short_ov) if short_ov else "")
                    )
                    lines.append(bullet)
                return self._with_source(f"{heading}:\n\n" + "\n\n".join(lines), "dataset")
            return self._with_source(recommend_no_director(canon), "dataset")
        return self._grounded_recommendation(user_query, retrieved)

    def _try_fast_answer(
        self,
        user_query: str,
        retrieved: list[dict],
        primary: Optional[dict],
    ) -> Optional[str]:
        """Dataset/web answers without the chat LLM (seconds, not minutes)."""
        if self._is_plot_or_about_question(user_query):
            if primary is not None:
                hit = self._plot_answer_from_primary(primary)
                if hit:
                    return hit
            spec = self._parse_query_spec(user_query)
            return self._answer_plot(user_query, retrieved, spec)

        if primary is not None and _is_opinion_question_text(user_query):
            person = self._extract_person_for_opinion(user_query)
            if person is None:
                return self._plot_answer_from_primary(primary)

        person = self._extract_person_for_opinion(user_query)
        if person:
            canon, movies, _corrected = resolve_director_movies(self.vs, person, limit=12)
            if movies:
                return self._format_director_opinion_answer(canon, movies)

        if self._is_recommend_query(user_query):
            return self._try_fast_recommend(user_query, retrieved)

        return None

    def _extract_title_for_web(self, user_query: str, spec: dict) -> str:
        raw = self._extract_movie_query_text(user_query)
        tokens = [
            t
            for t in self._norm(raw).split()
            if len(t) > 2
            and t
            not in {
                "movie",
                "film",
                "the",
                "tell",
                "what",
                "happens",
                "about",
                "please",
                "plot",
                "plaot",
                "story",
                "synopsis",
                "summary",
            }
        ]
        if tokens:
            return " ".join(tokens).title()
        compact = " ".join(spec.get("title_tokens", []))
        return compact.title() if compact else raw.strip()

    def _format_web_plot_answer(self, web: dict, title_fallback: str) -> str:
        return self._with_source(
            format_plot(
                str(web.get("title", title_fallback)),
                str(web.get("year") or "?"),
                str(web.get("overview", "")),
                str(web.get("director", "")),
            ),
            str(web.get("source", "web")),
        )

    def _collect_title_versions(
        self, title_for_web: str, retrieved: list[dict]
    ) -> list[dict]:
        """Distinct release years for the same film title (dataset + web)."""
        seen_years: set[int] = set()
        versions: list[dict] = []

        def _add(*, title: str, year: int, overview: str, director: str, source: str) -> None:
            if year in seen_years or not _same_film_title(title_for_web, title):
                return
            seen_years.add(year)
            ov = overview.strip()
            blurb = ov[:160].rstrip()
            if ov and len(ov) > 160:
                blurb += "..."
            versions.append(
                {
                    "title": title,
                    "year": year,
                    "director": director.strip(),
                    "blurb": blurb,
                    "source": source,
                }
            )

        for m in retrieved or []:
            y = _movie_record_year(m)
            if y is None:
                continue
            _add(
                title=str(m.get("title", title_for_web)),
                year=y,
                overview=str(m.get("overview", "")),
                director=str(m.get("director", "")),
                source="dataset",
            )

        for web in fetch_web_movie_candidates(title_for_web, limit=6):
            y = _extract_year_from_text(str(web.get("year", "")))
            if y is None:
                continue
            _add(
                title=str(web.get("title", title_for_web)),
                year=y,
                overview=str(web.get("overview", "")),
                director=str(web.get("director", "")),
                source=str(web.get("source", "web")),
            )

        versions.sort(key=lambda v: v["year"])
        return versions

    def _format_plot_disambiguation(self, title_display: str, versions: list[dict]) -> str:
        body: list[str] = []
        for i, v in enumerate(versions, start=1):
            director = v.get("director", "")
            dir_part = ""
            if director and director.lower() not in ("unknown", "unknown director"):
                first = director.split("|")[0].strip()
                if first:
                    dir_part = f" — {first}"
            body.append(f"{i}. {v['title']} ({v['year']}){dir_part}")
            if v.get("blurb"):
                body.append(f"   {v['blurb']}")
        return self._with_source(format_disambiguation(title_display, body), "dataset+web")

    def _title_tokens_match(self, spec: dict, title: str) -> bool:
        title_tokens = set(spec.get("title_tokens", []))
        if not title_tokens:
            return True
        t_tokens = set(self._norm(title).split())
        overlap = len(title_tokens & t_tokens)
        if overlap >= max(1, int(len(title_tokens) * 0.6)):
            return True
        title_for_web = " ".join(spec.get("title_tokens", []))
        return _same_film_title(title_for_web, title)

    def _resolve_film_context(
        self, user_query: str, retrieved: list[dict]
    ) -> tuple[Optional[dict], list[dict]]:
        """
        Pick the film the user means and augment retrieval with TMDB/web when
        the local catalogue has no match (e.g. asked year 2004, index only has 2007).
        """
        if self._is_recommend_query(user_query) or self._resolve_followup_director(user_query):
            primary = identify_primary_film(user_query, retrieved) if retrieved else None
            return primary, retrieved

        spec = self._parse_query_spec(user_query)
        title_for_web = self._extract_title_for_web(user_query, spec)
        asked_year = spec.get("year")
        has_title_hint = len(spec.get("title_tokens", [])) >= 2 or (
            len(title_for_web) >= 4
            and title_for_web.lower() not in ("his movies", "her movies", "their movies")
        )

        if asked_year is not None and has_title_hint:
            dataset_hit = None
            for m in retrieved:
                if not self._title_tokens_match(spec, str(m.get("title", ""))):
                    continue
                if _movie_record_year(m) == asked_year:
                    dataset_hit = m
                    break
            if dataset_hit is not None:
                return dataset_hit, retrieved

            web = fetch_web_movie_overview(title_for_web, year=asked_year)
            if web:
                primary = _web_hit_to_movie(web)
                filtered = [
                    m
                    for m in retrieved
                    if not (
                        _same_film_title(title_for_web, str(m.get("title", "")))
                        and _movie_record_year(m) is not None
                        and _movie_record_year(m) != asked_year
                    )
                ]
                return primary, [primary] + filtered[: TOP_K - 1]

        if has_title_hint:
            primary = identify_primary_film(user_query, retrieved)
            if primary is not None and self._title_tokens_match(spec, str(primary.get("title", ""))):
                py = _movie_record_year(primary)
                if asked_year is None or py is None or py == asked_year:
                    return primary, retrieved

            web = fetch_web_movie_overview(title_for_web, year=asked_year)
            if web and _same_film_title(title_for_web, str(web.get("title", ""))):
                primary = _web_hit_to_movie(web)
                return primary, [primary] + retrieved[: TOP_K - 1]

        primary = identify_primary_film(user_query, retrieved) if retrieved else None
        return primary, retrieved

    @staticmethod
    def _catalog_source_label(movie: Optional[dict]) -> str:
        if not movie:
            return "dataset"
        src = str(movie.get("catalog_source", "")).strip().lower()
        return src if src else "dataset"

    def _retrieve_movies(self, user_query: str, top_k: int = TOP_K) -> list[dict]:
        """Search with the full question and a title-focused variant."""
        merged: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def _add(results: list[dict]):
            for m in results:
                key = (str(m.get("title", "")), str(m.get("year", "")))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(m)

        _add(self.vs.search(user_query, top_k=top_k))
        title_text = self._extract_movie_query_text(user_query)
        year = self._extract_year(user_query)
        if title_text:
            clean = re.sub(
                r"\b(movie|movies|film|films|the|please|tell|me|what|happens|in|about)\b",
                " ",
                title_text,
                flags=re.IGNORECASE,
            )
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) >= 3:
                focused = f"{clean} {year}".strip() if year else clean
                if focused.lower() != user_query.lower():
                    _add(self.vs.search(focused, top_k=top_k))

        merged.sort(key=lambda m: float(m.get("relevance_score", 0.0)), reverse=True)
        return merged[:top_k]

    def _answer_plot(
        self, user_query: str, retrieved: list[dict], spec: dict
    ) -> Optional[str]:
        asked_year = spec.get("year")
        title_for_web = self._extract_title_for_web(user_query, spec)

        if asked_year is None:
            versions = self._collect_title_versions(title_for_web, retrieved or [])
            if len(versions) >= 2:
                display = title_for_web or versions[0]["title"]
                return self._format_plot_disambiguation(display, versions)

        if asked_year is not None:
            year_matches = [
                m
                for m in retrieved
                if _movie_record_year(m) == asked_year
                and float(m.get("relevance_score", 0.0)) >= 0.35
            ]
            if year_matches:
                best = identify_primary_film(user_query, year_matches) or year_matches[0]
                ov = str(best.get("overview", "")).strip()
                if ov:
                    return self._with_source(
                        format_plot(
                            str(best.get("title", title_for_web)),
                            str(best.get("year", asked_year)),
                            ov,
                            str(best.get("director", "")),
                        ),
                        "dataset",
                    )
            web = fetch_web_movie_overview(title_for_web, year=asked_year)
            if web:
                return self._format_web_plot_answer(web, title_for_web)

        if asked_year is None:
            web = fetch_web_movie_overview(title_for_web, year=None)
            if web and _same_film_title(title_for_web, str(web.get("title", ""))):
                return self._format_web_plot_answer(web, title_for_web)

        best = identify_primary_film(user_query, retrieved) if retrieved else None
        if best is not None:
            ov = str(best.get("overview", "")).strip()
            rel = float(best.get("relevance_score", 0.0))
            best_year = _movie_record_year(best)
            if asked_year is not None and best_year is not None and best_year != asked_year:
                best = None
            elif ov and rel >= 0.38:
                return self._with_source(
                    format_plot(
                        str(best.get("title", "Unknown")),
                        str(best.get("year", "?")),
                        ov,
                        str(best.get("director", "")),
                    ),
                    "dataset",
                )

        web = fetch_web_movie_overview(title_for_web, year=asked_year)
        if web:
            return self._format_web_plot_answer(web, title_for_web)

        if best is not None:
            ov = str(best.get("overview", "")).strip()
            if ov:
                return self._with_source(
                    format_plot(
                        str(best.get("title", "Unknown")),
                        str(best.get("year", "?")),
                        ov,
                        str(best.get("director", "")),
                    ),
                    "dataset",
                )
        return None

    def _wikidata_movie_fact(self, movie_query: str, expected_year: Optional[int] = None) -> Optional[dict]:
        try:
            def _get_json(url: str) -> dict:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "CineBot/1.0 (+local-rag-app)",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=8) as r:
                    return json.loads(r.read().decode("utf-8"))

            params = urllib.parse.urlencode(
                {
                    "action": "wbsearchentities",
                    "search": movie_query,
                    "language": "en",
                    "type": "item",
                    "format": "json",
                    "limit": 5,
                }
            )
            search_data = _get_json(f"https://www.wikidata.org/w/api.php?{params}")
            items = search_data.get("search", [])
            if not items:
                return None

            def _item_score(it: dict) -> float:
                desc = str(it.get("description", "")).lower()
                label = str(it.get("label", ""))
                score = 0.0
                if "film" in desc or "movie" in desc:
                    score += 5.0
                if expected_year is not None:
                    y1 = self._extract_year(desc)
                    y2 = self._extract_year(label)
                    if y1 == expected_year or y2 == expected_year:
                        score += 4.0
                q_tokens = set(self._norm(movie_query).split())
                l_tokens = set(self._norm(label).split())
                score += 0.2 * len(q_tokens & l_tokens)
                return score

            items = sorted(items, key=_item_score, reverse=True)[:5]

            def _fetch_claims(qid: str) -> dict:
                ent_params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": qid,
                        "languages": "en",
                        "format": "json",
                        "props": "claims",
                    }
                )
                ent_data = _get_json(f"https://www.wikidata.org/w/api.php?{ent_params}")
                return ent_data.get("entities", {}).get(qid, {}).get("claims", {})

            def _claim_entity_ids(claims: dict, prop: str, limit: int) -> list[str]:
                out = []
                for c in claims.get(prop, []):
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if isinstance(v, dict) and "id" in v:
                        out.append(v["id"])
                    if len(out) >= limit:
                        break
                return out

            def _claim_year(claims: dict, prop: str) -> Optional[int]:
                for c in claims.get(prop, []):
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if isinstance(v, dict):
                        t = str(v.get("time", ""))
                        y = self._extract_year(t)
                        if y is not None:
                            return y
                return None

            best = None
            best_score = -999.0
            for it in items:
                qid = it.get("id")
                if not qid:
                    continue
                claims = _fetch_claims(qid)
                cast_ids = _claim_entity_ids(claims, "P161", 5)
                dir_ids = _claim_entity_ids(claims, "P57", 2)
                rel_year = _claim_year(claims, "P577")

                score = _item_score(it)
                score += min(len(cast_ids), 3) * 1.5
                score += min(len(dir_ids), 2) * 0.7
                if expected_year is not None:
                    if rel_year == expected_year:
                        score += 5.0
                    elif rel_year is not None:
                        score -= 2.5

                if score > best_score:
                    best = (it, claims, cast_ids, dir_ids)
                    best_score = score

            if best is None:
                return None

            picked, _claims, cast_ids, dir_ids = best

            label_cache: dict[str, str] = {}

            def _label(eid: str) -> Optional[str]:
                if eid in label_cache:
                    return label_cache[eid]
                p = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": eid,
                        "languages": "en",
                        "format": "json",
                        "props": "labels",
                    }
                )
                dd = _get_json(f"https://www.wikidata.org/w/api.php?{p}")
                val = dd.get("entities", {}).get(eid, {}).get("labels", {}).get("en", {}).get("value")
                if val:
                    label_cache[eid] = val
                return val

            directors = [x for x in (_label(eid) for eid in dir_ids) if x]
            cast = [x for x in (_label(eid) for eid in cast_ids) if x]
            return {"title": picked.get("label", movie_query), "directors": directors, "cast": cast}
        except Exception:
            return None

    def _web_query_variants(self, user_query: str, spec: dict, movie: Optional[dict]) -> list[str]:
        base = self._extract_movie_query_text(user_query)
        franchise = re.search(
            r"\b(?:in|for|from|of)\s+(.+?)\s+(?:movies|films|series)\b",
            user_query,
            flags=re.IGNORECASE,
        )
        if franchise:
            hint = re.sub(
                r"\b(movie|movies|film|films|series)\b",
                "",
                franchise.group(1),
                flags=re.IGNORECASE,
            ).strip()
            if len(hint) >= 3:
                base = hint
        compact_title = " ".join(
            t for t in spec.get("title_tokens", []) if not t.isdigit() and t not in {"part", "one", "two"}
        ).strip()
        raw = base.lower()
        no_part = re.sub(r"\bpart\s*(one|two|1|2)\b", "", raw).strip()
        no_paren_year = re.sub(r"\(\s*(19\d{2}|20\d{2})\s*\)", "", no_part).strip()
        no_noise = re.sub(
            r"\b(who|what|is|the|actor|director|protagonist|main|lead|character|in|movie|film|tell|me)\b",
            " ",
            no_paren_year,
        )
        no_noise = re.sub(r"\s+", " ", no_noise).strip()
        year = spec.get("year")
        variants: list[str] = []
        for cand in (compact_title, base, raw, no_part, no_paren_year, no_noise):
            cand = cand.strip(" ?.,")
            if cand:
                variants.append(cand)
                if year is not None:
                    variants.append(f"{cand} {year}")
                    variants.append(f"{cand} ({year})")
        if movie is not None:
            t = str(movie.get("title", "")).strip()
            y = str(movie.get("year", "")).strip()
            if t:
                variants.append(t)
                if y:
                    variants.append(f"{t} ({y})")
        # Preserve order, dedupe, and skip tiny junk queries.
        out = []
        seen = set()
        for v in variants:
            key = self._norm(v)
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out[:3]

    def _wikidata_movie_fact_multi(self, queries: list[str], expected_year: Optional[int]) -> Optional[dict]:
        best = None
        best_score = -1
        for q in queries:
            fact = self._wikidata_movie_fact(q, expected_year=expected_year)
            if not fact:
                continue
            score = 0
            if fact.get("cast"):
                score += 2
            if fact.get("directors"):
                score += 1
            title = str(fact.get("title", ""))
            if expected_year is not None and self._extract_year(title) == expected_year:
                score += 2
            if score > best_score:
                best = fact
                best_score = score
        return best

    def _grounded_factual_answer(self, user_query: str, retrieved: list[dict]) -> Optional[str]:
        if not retrieved:
            return None
        q = self._norm(user_query)
        spec = self._parse_query_spec(user_query)
        movie, confident = self._select_factual_candidate(user_query, retrieved)

        if spec["asks_plot"] or spec.get("asks_multi_plot"):
            if spec.get("asks_multi_plot"):
                lines = []
                seen_titles: set[str] = set()
                for m in retrieved[:6]:
                    t = str(m.get("title", ""))
                    if not t or t in seen_titles:
                        continue
                    seen_titles.add(t)
                    ov = str(m.get("overview", "")).strip()
                    if not ov:
                        continue
                    lines.append(
                        f"• {t} ({m.get('year', '?')}): {ov[:320].rstrip()}"
                        + ("..." if len(ov) > 320 else "")
                    )
                if lines:
                    return self._with_source(
                        multi_plot_intro() + "\n\n".join(lines),
                        "dataset",
                    )
            plot_answer = self._answer_plot(user_query, retrieved, spec)
            if plot_answer:
                return plot_answer

        web_fact = None
        needs_web = (
            spec["asks_actor"]
            or spec["asks_protagonist"]
            or spec["asks_character"]
            or spec["asks_director"]
        )
        if needs_web and (movie is None or not confident or spec["asks_actor"] or spec["asks_protagonist"]):
            queries = self._web_query_variants(user_query, spec, movie)
            web_fact = self._wikidata_movie_fact_multi(queries, expected_year=spec.get("year"))

        title = str(movie.get("title", "that movie")) if movie is not None else "that movie"
        year = str(movie.get("year", "?")) if movie is not None else "?"
        director = str(movie.get("director", "unknown")) if movie is not None else "unknown"
        genre = str(movie.get("genre", "unknown")) if movie is not None else "unknown"
        rating = str(movie.get("rating", "unknown")) if movie is not None else "unknown"
        overview = str(movie.get("overview", "")).strip() if movie is not None else ""

        if spec["asks_director"]:
            if movie is None and web_fact and web_fact.get("directors"):
                return self._with_source(
                    format_director_web(
                        str(web_fact["title"]),
                        ", ".join(web_fact["directors"]),
                    ),
                    "web",
                )
            if movie is None or not confident:
                if web_fact and web_fact.get("directors"):
                    return self._with_source(
                        format_director_web(
                            str(web_fact["title"]),
                            ", ".join(web_fact["directors"]),
                        ),
                        "web",
                    )
                return self._abstain(user_query)
            return self._with_source(format_director(title, year, director), "dataset")
        if spec["asks_year"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(format_year(title, year), "dataset")
        if spec["asks_character"]:
            char_name = str(spec.get("character_name", "")).strip()
            char_name = re.sub(
                r"\bthe\s+(grey|gray|white|dark)\b",
                "",
                char_name,
                flags=re.IGNORECASE,
            ).strip()
            film_hint = str(spec.get("character_film", "")).strip()
            film_hint = re.sub(r"\b(19\d{2}|20\d{2})\b.*$", "", film_hint).strip()
            film_hint = re.sub(r"\b(movie|film|movies|films)\b", "", film_hint, flags=re.IGNORECASE).strip()
            actor = self._wikidata_actor_from_film_cast(
                film_hint,
                char_name,
                expected_year=spec.get("year"),
                retrieved=retrieved,
            )
            if not actor:
                actor = self._wikidata_character_portrayal(char_name, film_hint)
            film_label = title if movie is not None else (web_fact or {}).get("title", film_hint or "that film")
            if actor:
                return self._with_source(
                    format_character_portrayal(char_name, film_label, actor),
                    "web",
                )
            return self._abstain(user_query)
        if spec["asks_actor"]:
            if web_fact and web_fact.get("cast"):
                lead = web_fact["cast"][0]
                tail = ", ".join(web_fact["cast"][1:4])
                film_label = str(web_fact.get("title", title))
                if re.search(r"\b(movies|films|series)\b", user_query, flags=re.IGNORECASE):
                    return self._with_source(
                        format_cast_lead(film_label, lead, tail),
                        "web",
                    )
                return self._with_source(
                    format_cast_lead(film_label, lead, tail),
                    "web",
                )
            return self._abstain(user_query)
        if spec["asks_protagonist"]:
            if web_fact and web_fact.get("cast"):
                return self._with_source(
                    format_protagonist(web_fact["cast"][0], str(web_fact["title"])),
                    "web",
                )
            if overview:
                return self._with_source(
                    format_plot(title, year, overview[:220].rstrip() + "..."),
                    "dataset",
                )
            return self._abstain(user_query)
        if spec["asks_genre"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(format_genre(title, year, genre), "dataset")
        if spec["asks_rating"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(format_rating(title, year, rating), "dataset")
        if spec["asks_plot"]:
            plot_answer = self._answer_plot(user_query, retrieved, spec)
            if plot_answer:
                return plot_answer
            if movie is None or not confident:
                if overview:
                    return self._with_source(
                        format_plot(title, year, overview, director),
                        "dataset",
                    )
                return self._abstain(user_query)
            if overview:
                return self._with_source(
                    format_plot(title, year, overview, director),
                    "dataset",
                )
            return self._with_source(format_no_plot(title), "dataset")
        return None

    @staticmethod
    def _looks_like_director_name(phrase: str) -> bool:
        """Reject cast/role phrases mistaken for director names."""
        if is_cast_or_actor_question(phrase):
            return False
        tokens = phrase.lower().split()
        if len(tokens) < 2 or len(tokens) > 5:
            return False
        blocked = {
            "lead",
            "role",
            "actor",
            "actress",
            "cast",
            "star",
            "stars",
            "starring",
            "character",
            "protagonist",
            "hero",
            "villain",
            "plays",
            "played",
            "portrays",
            "portrayed",
            "has",
            "had",
            "in",
            "main",
            "supporting",
            "movies",
            "movie",
            "films",
            "film",
            "series",
        }
        if any(t in blocked for t in tokens):
            return False
        return True

    def _extract_director_constraint(self, user_query: str) -> Optional[str]:
        if is_cast_or_actor_question(user_query):
            return None
        if self._is_plot_or_about_question(user_query):
            return None
        qn = re.sub(r"[^a-z0-9 ]+", " ", user_query.lower())
        qn = re.sub(
            r"\b(suggest|sugget|recommend|recomend|find|give|show|please|some|any|me|top|rated|best|good|great|highest|what|are|the|which|who|is|was|were|a|an)\b",
            " ",
            qn,
        )
        qn = re.sub(r"\s+", " ", qn).strip()

        for pat in (
            r"\b(?:movies|films|flicks)\s+(?:by|from|of)\s+([a-z]+(?:\s+[a-z]+)+)\b",
            r"(?<!plot )(?<!story )(?<!synopsis )(?<!summary )\b(?:by|from)\s+([a-z]+(?:\s+[a-z]+)+)\b",
            r"\b(?:directed by|director)\s+([a-z]+(?:\s+[a-z]+)+)\b",
            r"\b([a-z]+(?:\s+[a-z]+)+)\s+(?:movies|films|flicks)\b",
        ):
            m = re.search(pat, qn)
            if m:
                name = m.group(1).strip()
                if self._looks_like_director_name(name):
                    return name
        return None

    @staticmethod
    def _wants_top_rated(user_query: str) -> bool:
        qn = user_query.lower()
        return any(
            sig in qn
            for sig in ("top rated", "best rated", "highest rated", "top ", "best ")
        )

    def _grounded_recommendation(self, user_query: str, retrieved: list[dict]) -> Optional[str]:
        director = self._extract_director_constraint(user_query)
        corrected = False
        if director:
            canon, retrieved, corrected = resolve_director_movies(self.vs, director, limit=12)
            director = canon
            if not retrieved:
                return self._with_source(recommend_no_director(director), "dataset")
            heading = recommend_heading_director(director, corrected)
        elif not retrieved:
            return recommend_no_matches()
        else:
            heading = recommend_heading_general()

        if director or self._wants_top_rated(user_query):
            retrieved = sorted(
                retrieved,
                key=lambda m: _parse_rating_value(m.get("rating", "")),
                reverse=True,
            )

        lines = []
        for i, m in enumerate(retrieved[:5], start=1):
            title = str(m.get("title", "Unknown title"))
            year = str(m.get("year", "?"))
            director_name = str(m.get("director", "Unknown director"))
            genre = str(m.get("genre", "Unknown genre"))
            rating = str(m.get("rating", "?"))
            score = float(m.get("relevance_score", 0.0))
            overview = str(m.get("overview", "")).strip()
            short_ov = overview[:180].rstrip()
            if short_ov and len(overview) > 180:
                short_ov += "..."
            if director:
                bullet = f"{i}. {title} ({year}) — {director_name} [{genre}] · rating {rating}"
            else:
                bullet = f"{i}. {title} ({year}) — {director_name} [{genre}] · match {score:.3f}"
            if short_ov:
                bullet += recommend_bullet_why(short_ov)
            lines.append(bullet)

        return self._with_source(f"{heading}:\n\n" + "\n\n".join(lines), "dataset")

    @staticmethod
    def _looks_garbled(text: str) -> bool:
        if not text:
            return True
        weird_chars = sum(
            1 for ch in text if ord(ch) > 127 and ch not in "éáíóúñü—–•"
        )
        weird_ratio = weird_chars / max(1, len(text))
        replacement_ratio = text.count("�") / max(1, len(text))
        if weird_ratio > 0.12 or replacement_ratio > 0.01:
            return True
        tokens = re.findall(r"[a-z]{3,}", text.lower())
        if len(tokens) >= 12:
            from collections import Counter

            common, count = Counter(tokens).most_common(1)[0]
            if count / len(tokens) > 0.22 and len(common) <= 6:
                return True
        if re.search(r"\bquant\b", text.lower()) and text.lower().count("quant") >= 4:
            return True
        return False

    def _wikidata_resolve_film_qid(
        self, film_search: str, expected_year: Optional[int] = None
    ) -> Optional[str]:
        try:
            params = urllib.parse.urlencode(
                {
                    "action": "wbsearchentities",
                    "search": film_search,
                    "language": "en",
                    "type": "item",
                    "format": "json",
                    "limit": 6,
                }
            )
            data = _http_json(f"https://www.wikidata.org/w/api.php?{params}")
            items = data.get("search", [])
            if not items:
                return None

            def _score(it: dict) -> float:
                desc = str(it.get("description", "")).lower()
                label = str(it.get("label", "")).lower()
                score = 0.0
                if "film" in desc or "movie" in desc:
                    score += 5.0
                if expected_year is not None:
                    y = self._extract_year(desc) or self._extract_year(label)
                    if y == expected_year:
                        score += 4.0
                return score

            items = sorted(items, key=_score, reverse=True)
            return items[0].get("id")
        except Exception:
            return None

    def _wikidata_actor_from_film_cast(
        self,
        film_hint: str,
        character: str,
        *,
        expected_year: Optional[int] = None,
        retrieved: Optional[list[dict]] = None,
    ) -> Optional[str]:
        char_key = self._norm(character)
        if len(char_key) < 3:
            return None
        film_queries: list[str] = []
        if film_hint:
            film_queries.append(film_hint)
        if retrieved:
            for m in retrieved[:2]:
                t = str(m.get("title", "")).strip()
                y = str(m.get("year", "")).strip()
                if t:
                    film_queries.append(t)
                    if y and y != "Unknown":
                        film_queries.append(f"{t} ({y})")
        seen_q: set[str] = set()
        for fq in film_queries[:3]:
            key = self._norm(fq)
            if key in seen_q:
                continue
            seen_q.add(key)
            qid = self._wikidata_resolve_film_qid(fq, expected_year=expected_year)
            if not qid:
                continue
            try:
                ent_params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": qid,
                        "languages": "en",
                        "format": "json",
                        "props": "claims",
                    }
                )
                ent = _http_json(f"https://www.wikidata.org/w/api.php?{ent_params}")
                cast_claims = ent.get("entities", {}).get(qid, {}).get("claims", {}).get("P161", [])
                role_ids: list[str] = []
                actor_for_role: dict[str, str] = {}
                for c in cast_claims:
                    actor_val = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if not isinstance(actor_val, dict) or "id" not in actor_val:
                        continue
                    actor_id = actor_val["id"]
                    for qual in c.get("qualifiers", {}).get("P453", []):
                        role_val = qual.get("datavalue", {}).get("value", {})
                        if isinstance(role_val, dict) and "id" in role_val:
                            rid = role_val["id"]
                            role_ids.append(rid)
                            actor_for_role[rid] = actor_id
                if not role_ids:
                    continue
                label_params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": "|".join(role_ids[:40]),
                        "languages": "en",
                        "format": "json",
                        "props": "labels",
                    }
                )
                labels_data = _http_json(f"https://www.wikidata.org/w/api.php?{label_params}")
                for rid in role_ids:
                    role_label = (
                        labels_data.get("entities", {})
                        .get(rid, {})
                        .get("labels", {})
                        .get("en", {})
                        .get("value", "")
                    )
                    if not role_label:
                        continue
                    if char_key in self._norm(role_label) or self._norm(role_label) in char_key:
                        actor_id = actor_for_role.get(rid)
                        if not actor_id:
                            continue
                        actor_label = (
                            _http_json(
                                "https://www.wikidata.org/w/api.php?"
                                + urllib.parse.urlencode(
                                    {
                                        "action": "wbgetentities",
                                        "ids": actor_id,
                                        "languages": "en",
                                        "format": "json",
                                        "props": "labels",
                                    }
                                )
                            )
                            .get("entities", {})
                            .get(actor_id, {})
                            .get("labels", {})
                            .get("en", {})
                            .get("value")
                        )
                        if actor_label:
                            return actor_label
            except Exception:
                continue
        return None

    def _wikidata_character_portrayal(self, character: str, film_hint: str = "") -> Optional[str]:
        try:
            search = f"{character} {film_hint}".strip() if film_hint else character
            params = urllib.parse.urlencode(
                {
                    "action": "wbsearchentities",
                    "search": search,
                    "language": "en",
                    "type": "item",
                    "format": "json",
                    "limit": 8,
                }
            )
            data = _http_json(f"https://www.wikidata.org/w/api.php?{params}")
            items = data.get("search", [])
            if not items:
                return None

            def _score_item(it: dict) -> float:
                desc = str(it.get("description", "")).lower()
                label = str(it.get("label", "")).lower()
                score = 0.0
                if "character" in desc or "fictional" in desc:
                    score += 6.0
                if character.lower() in label:
                    score += 4.0
                if film_hint:
                    fh = self._norm(film_hint)
                    if any(tok in label or tok in desc for tok in fh.split() if len(tok) > 3):
                        score += 2.0
                return score

            items = sorted(items, key=_score_item, reverse=True)
            for it in items[:5]:
                qid = it.get("id")
                if not qid:
                    continue
                ent_params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": qid,
                        "languages": "en",
                        "format": "json",
                        "props": "claims",
                    }
                )
                ent = _http_json(f"https://www.wikidata.org/w/api.php?{ent_params}")
                claims = ent.get("entities", {}).get(qid, {}).get("claims", {})
                actor_ids: list[str] = []
                for c in claims.get("P175", []):
                    v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    if isinstance(v, dict) and "id" in v:
                        actor_ids.append(v["id"])
                if not actor_ids:
                    continue
                label_params = urllib.parse.urlencode(
                    {
                        "action": "wbgetentities",
                        "ids": "|".join(actor_ids[:3]),
                        "languages": "en",
                        "format": "json",
                        "props": "labels",
                    }
                )
                labels_data = _http_json(f"https://www.wikidata.org/w/api.php?{label_params}")
                for eid in actor_ids:
                    name = (
                        labels_data.get("entities", {})
                        .get(eid, {})
                        .get("labels", {})
                        .get("en", {})
                        .get("value")
                    )
                    if name:
                        return name
        except Exception:
            return None
        return None

    def _grounded_discussion(
        self,
        retrieved: list[dict],
        user_query: str = "",
        primary: Optional[dict] = None,
    ) -> str:
        if not retrieved and primary is None:
            return recommend_no_matches()
        top = primary
        if top is None:
            top = identify_primary_film(user_query, retrieved) if user_query else None
        if top is None and retrieved:
            top = retrieved[0]
        if top is None:
            return recommend_no_matches()
        title = str(top.get("title", "Unknown title"))
        year = str(top.get("year", "?"))
        director = str(top.get("director", "Unknown director"))
        genre = str(top.get("genre", "Unknown genre"))
        overview = str(top.get("overview", "")).strip()
        source = self._catalog_source_label(top)
        if overview:
            body = format_plot(title, year, overview[:260] + ("..." if len(overview) > 260 else ""), director)
        else:
            body = (
                f'Closest match in your catalogue: "{title}" ({year}) — {director} [{genre}]. '
                f"No synopsis on file, but the title's there if you want to dig in."
            )
        return body + f"\n\nSource used: {source}"

    @staticmethod
    def _is_simple_movie_lookup(user_query: str) -> bool:
        qn = re.sub(r"[^a-z0-9 ]+", " ", user_query.lower()).strip()
        words = qn.split()
        if not words or len(words) > 12:
            return False
        blocked = (
            "who ",
            "what ",
            "when ",
            "where ",
            "why ",
            "how ",
            "which ",
            "recommend",
            "suggest",
            "director",
            "actor",
            "rating",
            "genre",
            "compare",
            " vs ",
            "versus",
        )
        return not any(b in qn for b in blocked)

    def respond(self, user_query: str) -> tuple[str, list[dict]]:
        user_query = normalize_user_query(user_query)

        # Recommend / "more of his movies" — skip web lookup and LLM entirely.
        if self._is_recommend_query(user_query) or self._resolve_followup_director(user_query):
            retrieved = self._retrieve_movies(user_query, top_k=TOP_K)
            fast_rec = self._try_fast_recommend(user_query, retrieved)
            if fast_rec:
                self.history.append({"role": "user", "content": user_query})
                self.history.append({"role": "assistant", "content": fast_rec})
                return fast_rec, retrieved

        retrieved = self._retrieve_movies(user_query, top_k=TOP_K)
        primary, retrieved = self._resolve_film_context(user_query, retrieved)
        context = self.vs.format_context(retrieved)
        primary_source = self._catalog_source_label(primary)

        fast = self._try_fast_answer(user_query, retrieved, primary)
        if fast is not None:
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": fast})
            return fast, retrieved

        intent = detect_intent(user_query)
        if self._is_plot_or_about_question(user_query):
            intent = "factual"
        director_filter = self._extract_director_constraint(user_query)
        if (
            director_filter
            and not is_cast_or_actor_question(user_query)
            and not self._is_plot_or_about_question(user_query)
        ):
            intent = "recommend"

        direct = None
        if intent == "factual":
            direct = self._grounded_factual_answer(user_query, retrieved)
        elif intent == "recommend":
            if director_filter:
                _canon, retrieved, _fixed = resolve_director_movies(self.vs, director_filter, limit=12)
                primary, retrieved = self._resolve_film_context(user_query, retrieved)
                context = self.vs.format_context(retrieved)
            direct = self._grounded_recommendation(user_query, retrieved)
        elif intent == "discussion" and retrieved and self._is_simple_movie_lookup(user_query):
            best = primary or (identify_primary_film(user_query, retrieved) if user_query else None)
            if best is None and retrieved:
                best = retrieved[0]
            ov = str(best.get("overview", "")).strip() if best else ""
            rel = float(best.get("relevance_score", 0.0)) if best else 0.0
            if best and ov and rel >= 0.45:
                direct = self._with_source(
                    format_plot(
                        str(best.get("title", "Unknown")),
                        str(best.get("year", "?")),
                        ov,
                        str(best.get("director", "")),
                    ),
                    primary_source,
                )
        elif self._is_plot_or_about_question(user_query):
            spec = self._parse_query_spec(user_query)
            direct = self._answer_plot(user_query, retrieved, spec)
        if direct is not None:
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": direct})
            return direct, retrieved

        if self.use_generative:
            response = self._respond_generative(
                user_query,
                retrieved,
                intent=intent,
                primary=primary,
                primary_source=primary_source,
                context=context,
            )
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": response})
            return response, retrieved

        fallback = self._deterministic_fallback(
            user_query, retrieved, intent=intent, primary=primary, director_filter=director_filter
        )
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": fallback})
        return fallback, retrieved

    def _deterministic_fallback(
        self,
        user_query: str,
        retrieved: list[dict],
        *,
        intent: str,
        primary: Optional[dict],
        director_filter: Optional[str],
    ) -> str:
        """Always-available fast answers (no LLM)."""
        again = self._try_fast_answer(user_query, retrieved, primary)
        if again:
            return again

        if intent == "recommend":
            if director_filter:
                _canon, retrieved, _fixed = resolve_director_movies(self.vs, director_filter, limit=12)
            hit = self._try_fast_recommend(user_query, retrieved)
            if hit:
                return hit

        if intent == "factual":
            hit = self._grounded_factual_answer(user_query, retrieved)
            if hit:
                return hit

        hit = self._grounded_discussion(retrieved, user_query, primary=primary)
        if hit:
            return hit

        return (
            "I couldn't line that up in your catalogue fast enough — "
            "try a film title with a year, a director name, or 'suggest movies by Christopher Nolan'."
        )

    def _respond_generative(
        self,
        user_query: str,
        retrieved: list[dict],
        *,
        intent: str,
        primary: Optional[dict],
        primary_source: str,
        context: str,
    ) -> str:
        """Optional slow path — only when use_generative=True."""
        history_text = ""
        for turn in self.history[-6:]:
            history_text += f"\n{turn['role'].capitalize()}: {turn['content']}"

        user_block = RAG_USER_BODY.format(context=context, query=user_query)
        if primary is not None and str(primary.get("catalog_source", "")) not in ("", "dataset"):
            user_block = (
                "NOTE: The user's film is verified from TMDB/Wikipedia (not in the local CSV). "
                "The first CONTEXT entry is authoritative — do not substitute a different year or remake.\n\n"
                + user_block
            )
        if history_text.strip():
            user_block = f"(Earlier conversation:{history_text})\n\n{user_block}"

        if self.prompt_format == "phi3":
            prompt = format_phi3_prompt(SYSTEM_PROMPT, user_block)
        else:
            prompt = format_raw_prompt(SYSTEM_PROMPT, user_block)

        response = self.model.generate(prompt)  # type: ignore[union-attr]
        if self._looks_garbled(response):
            return self._deterministic_fallback(
                user_query, retrieved, intent=intent, primary=primary, director_filter=None
            )
        if primary is not None and primary_source not in ("dataset", "") and "Source used:" not in response:
            response = response.rstrip() + f"\n\nSource used: {primary_source}"
        return response

    def reset(self):
        self.history.clear()


def main():
    parser = argparse.ArgumentParser(description="Movie Nerd RAG")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--recommend", type=str, default=None)
    parser.add_argument("--movies", default="databse.csv")
    parser.add_argument("--model", default=None)
    parser.add_argument("--store", default=CHROMA_DIR)
    parser.add_argument("--vector-backend", choices=["auto", "chroma", "numpy"], default="auto")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--prompt-format", choices=["phi3", "raw"], default="phi3")
    args = parser.parse_args()

    vs = MovieVectorStore(persist_dir=args.store, backend=args.vector_backend)
    if args.vector_backend == "auto":
        print(f"Vector backend: {vs.backend}")

    if args.build:
        print(f"Building index from {args.movies}...")
        vs.build(args.movies)
        if not args.chat and not args.recommend:
            return

    model = None
    if args.model:
        p = Path(args.model)
        if not p.exists():
            print(f"Model not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        if p.is_dir():
            model = HFMovieNerdModel(str(p))
        else:
            model = MovieNerdModel(str(p), n_gpu_layers=args.gpu_layers)
    elif args.chat:
        print("No --model: chat runs in retrieval-only mode.\n")

    chat = MovieNerdChat(vs, model, prompt_format=args.prompt_format)

    if args.recommend:
        text, movies = chat.respond(args.recommend)
        print(f"\nMr. Cinephile: {text}\n\nRetrieved:")
        for m in movies:
            print(f"  [{m['relevance_score']:.2f}] {m['title']} ({m.get('year', '?')}) — {m.get('director', '?')}")
        return

    if args.chat:
        print("\n" + "═" * 56)
        print("  Mr. Cinephile — grounded film enthusiast (RAG)")
        print("  quit | reset | sources")
        print("═" * 56 + "\n")
        last_retrieved: list[dict] = []
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_input:
                continue
            low = user_input.lower()
            if low in ("quit", "exit", "bye"):
                print("Mr. Cinephile: Cheers — happy viewing!")
                break
            if low == "reset":
                chat.reset()
                continue
            if low == "sources":
                if last_retrieved:
                    for m in last_retrieved:
                        print(f"  [{m['relevance_score']:.2f}] {m['title']} ({m.get('year', '?')})")
                else:
                    print("  (none yet)")
                continue

            print("Mr. Cinephile: ", end="", flush=True)
            reply, last_retrieved = chat.respond(user_input)
            print(reply)
            print()

    if not args.build and not args.chat and not args.recommend:
        parser.print_help()


if __name__ == "__main__":
    main()
