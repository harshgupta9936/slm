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
import re
import sys
import urllib.parse
import urllib.request
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

from movie_data import load_movies_dataframe

CHROMA_DIR = "./movie_vector_store"
COLLECTION = "movies"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
CONTEXT_WINDOW = 4096

SYSTEM_PROMPT = """You are CinéBot, a passionate and opinionated movie nerd. You speak like a \
real film enthusiast — casual, enthusiastic, with strong opinions. You MUST ground every claim \
about specific movies, directors, years, genres, and plots in the CONTEXT block below. If the \
context does not mention a movie or fact, say you do not have it in your database instead of \
guessing. You may still share general film opinions that do not assert database-specific facts."""

RAG_USER_BODY = """CONTEXT (only use these films for factual claims):
{context}

User question:
{query}

Answer as CinéBot. For facts about films listed above, stick to the context."""


def format_phi3_prompt(system: str, user: str) -> str:
    """Phi-3 / Phi-3.5 style markers (works with common GGUF chat templates)."""
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n"
    )


def format_raw_prompt(system: str, user: str) -> str:
    return f"{system}\n\n{user}\n\nCinéBot:"


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


class _ChromaMovieStore:
    def __init__(self, persist_dir: str):
        if not _HAS_CHROMA:
            raise RuntimeError("chromadb is not installed")
        self.persist_dir = persist_dir
        self.embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device="cpu",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)

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


class MovieVectorStore:
    """auto: Chroma if installed, else NumPy + MiniLM."""

    def __init__(self, persist_dir: str = CHROMA_DIR, backend: str = "auto"):
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

    @staticmethod
    def format_context(movies: list[dict]) -> str:
        lines = []
        for m in movies:
            lines.append(
                f"• {m['title']} ({m.get('year', '?')}) — {m.get('director', '?')} "
                f"[{m.get('genre', '?')}]\n  {m.get('overview', '')[:400]}"
            )
        return "\n\n".join(lines)


def detect_intent(query: str) -> str:
    q = query.lower()
    q_simple = re.sub(r"[^a-z0-9 ]+", " ", q)
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
        "tell me about",
    ):
        if sig in q:
            return "factual"
    return "discussion"


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
    ):
        self.vs = vector_store
        self.model = model
        self.history: list[dict] = []
        self.prompt_format = prompt_format

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
        m = re.search(r"\b(?:of|for|about|in)\s+(.+)$", user_query, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" ?.")
        q = re.sub(
            r"^\s*(tell me|please|can you|could you|who is|what is|what are|give me)\s+",
            "",
            user_query,
            flags=re.IGNORECASE,
        )
        return q.strip(" ?.")

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
        asks_actor = any(
            sig in qn
            for sig in ("actor", "lead actor", "main actor", "cast", "starring")
        )
        asks_protagonist = any(sig in qn for sig in ("protagonist", "main character", "hero of"))
        asks_director = any(sig in qn for sig in ("director", "who directed", "who made"))
        asks_year = any(sig in qn for sig in ("what year", "when was", "release year", "came out"))
        asks_plot = any(sig in qn for sig in ("plot", "summary", "tell me about", "about"))
        asks_genre = any(sig in qn for sig in ("genre", "what kind of movie"))
        asks_rating = any(sig in qn for sig in ("rating", "score", "how good"))
        return {
            "year": self._extract_year(user_query),
            "part": self._part_hint(user_query),
            "title_tokens": title_tokens,
            "asks_actor": asks_actor,
            "asks_protagonist": asks_protagonist,
            "asks_director": asks_director,
            "asks_year": asks_year,
            "asks_plot": asks_plot,
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
            score = rel + overlap * 0.08

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
        return self._with_source(
            f"I can't verify a high-confidence answer for '{user_query}' from my grounded sources, so I won't guess.",
            source,
        )

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
        return out[:8]

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
        web_fact = None
        if movie is None or not confident or spec["asks_actor"] or spec["asks_protagonist"]:
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
                    f"{web_fact['title']} was directed by {', '.join(web_fact['directors'])}.",
                    "web",
                )
            if movie is None or not confident:
                if web_fact and web_fact.get("directors"):
                    return self._with_source(
                        f"{web_fact['title']} was directed by {', '.join(web_fact['directors'])}.",
                        "web",
                    )
                return self._abstain(user_query)
            return self._with_source(f"{title} ({year}) was directed by {director}.", "dataset")
        if spec["asks_year"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(f"{title} was released in {year}.", "dataset")
        if spec["asks_actor"]:
            if web_fact and web_fact.get("cast"):
                lead = web_fact["cast"][0]
                tail = ", ".join(web_fact["cast"][1:4])
                if tail:
                    return self._with_source(
                        f"{web_fact['title']} stars {lead}; also featuring {tail}.",
                        "web",
                    )
                return self._with_source(f"{web_fact['title']} stars {lead}.", "web")
            return self._abstain(user_query)
        if spec["asks_protagonist"]:
            if web_fact and web_fact.get("cast"):
                return self._with_source(
                    f"The protagonist is portrayed by {web_fact['cast'][0]} in {web_fact['title']}.",
                    "web",
                )
            if overview:
                return self._with_source(
                    f"From your dataset summary of {title} ({year}), the story centers on: {overview[:220].rstrip()}...",
                    "dataset",
                )
            return self._abstain(user_query)
        if spec["asks_genre"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(f"{title} ({year}) is listed as {genre}.", "dataset")
        if spec["asks_rating"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            return self._with_source(f"{title} ({year}) has a rating of {rating} in your dataset.", "dataset")
        if spec["asks_plot"]:
            if movie is None or not confident:
                return self._abstain(user_query)
            if overview:
                return self._with_source(f"{title} ({year}): {overview}", "dataset")
            return self._with_source(f"I don't have a plot summary for {title} in your dataset.", "dataset")
        return None

    def _grounded_recommendation(self, user_query: str, retrieved: list[dict]) -> Optional[str]:
        if not retrieved:
            return "I couldn't find close matches in your movie dataset for that request."

        lines = []
        for i, m in enumerate(retrieved[:5], start=1):
            title = str(m.get("title", "Unknown title"))
            year = str(m.get("year", "?"))
            director = str(m.get("director", "Unknown director"))
            genre = str(m.get("genre", "Unknown genre"))
            score = float(m.get("relevance_score", 0.0))
            overview = str(m.get("overview", "")).strip()
            short_ov = overview[:180].rstrip()
            if short_ov and len(overview) > 180:
                short_ov += "..."
            bullet = f"{i}. {title} ({year}) — {director} [{genre}] · match {score:.3f}"
            if short_ov:
                bullet += f"\n   Why: {short_ov}"
            lines.append(bullet)

        return self._with_source("Top grounded picks from your dataset:\n\n" + "\n\n".join(lines), "dataset")

    @staticmethod
    def _looks_garbled(text: str) -> bool:
        if not text:
            return True
        weird_chars = sum(
            1 for ch in text if ord(ch) > 127 and ch not in "éáíóúñü—–•"
        )
        weird_ratio = weird_chars / max(1, len(text))
        replacement_ratio = text.count("�") / max(1, len(text))
        return weird_ratio > 0.12 or replacement_ratio > 0.01

    def _grounded_discussion(self, retrieved: list[dict]) -> str:
        if not retrieved:
            return "I couldn't find relevant movies in your dataset for that."
        top = retrieved[0]
        title = str(top.get("title", "Unknown title"))
        year = str(top.get("year", "?"))
        director = str(top.get("director", "Unknown director"))
        genre = str(top.get("genre", "Unknown genre"))
        overview = str(top.get("overview", "")).strip()
        snippet = overview[:260].rstrip()
        if snippet and len(overview) > 260:
            snippet += "..."
        return (
            f"Closest grounded match: {title} ({year}) by {director} [{genre}].\n"
            f"{snippet if snippet else 'No overview available in your dataset.'}"
        ) + "\n\nSource used: dataset"

    def respond(self, user_query: str) -> tuple[str, list[dict]]:
        intent = detect_intent(user_query)
        retrieved = self.vs.search(user_query, top_k=TOP_K)
        context = self.vs.format_context(retrieved)

        direct = None
        if intent == "factual":
            direct = self._grounded_factual_answer(user_query, retrieved)
        elif intent == "recommend":
            direct = self._grounded_recommendation(user_query, retrieved)
        if direct is not None:
            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": direct})
            return direct, retrieved

        if self.model is None:
            return (
                "[No model loaded — retrieval only]\n\n" + context,
                retrieved,
            )

        history_text = ""
        for turn in self.history[-6:]:
            history_text += f"\n{turn['role'].capitalize()}: {turn['content']}"

        user_block = RAG_USER_BODY.format(context=context, query=user_query)
        if history_text.strip():
            user_block = f"(Earlier conversation:{history_text})\n\n{user_block}"

        if self.prompt_format == "phi3":
            prompt = format_phi3_prompt(SYSTEM_PROMPT, user_block)
        else:
            prompt = format_raw_prompt(SYSTEM_PROMPT, user_block)

        response = self.model.generate(prompt)
        if self._looks_garbled(response):
            # Last-resort safety: return deterministic grounded text instead of gibberish.
            if intent == "recommend":
                response = self._grounded_recommendation(user_query, retrieved) or self._grounded_discussion(retrieved)
            elif intent == "factual":
                response = self._grounded_factual_answer(user_query, retrieved) or self._grounded_discussion(retrieved)
            else:
                response = self._grounded_discussion(retrieved)
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": response})
        return response, retrieved

    def reset(self):
        self.history.clear()
        print("  [Conversation reset]")


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
        print(f"\nCinéBot: {text}\n\nRetrieved:")
        for m in movies:
            print(f"  [{m['relevance_score']:.2f}] {m['title']} ({m.get('year', '?')}) — {m.get('director', '?')}")
        return

    if args.chat:
        print("\n" + "═" * 56)
        print("  CinéBot — grounded movie nerd (RAG)")
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
                print("CinéBot: Later!")
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

            print("CinéBot: ", end="", flush=True)
            reply, last_retrieved = chat.respond(user_input)
            print(reply)
            print()

    if not args.build and not args.chat and not args.recommend:
        parser.print_help()


if __name__ == "__main__":
    main()
