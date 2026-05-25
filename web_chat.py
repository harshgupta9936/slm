"""

Local web UI for CinéBot (RAG + optional GGUF).

Run:
  python web_chat.py --model .\\movie-nerd.Q4_K_M.gguf --gpu-layers 20

Or (recommended if GGUF generation looks like gibberish):
  python web_chat.py --model .\\movie-nerd-lora\\merged-model

Then open:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from importlib.util import module_from_spec, spec_from_file_location

_ROOT = Path(__file__).resolve().parent
_UI_DIR = _ROOT / "webui"


def _load_rag_module():
    spec = spec_from_file_location("rag_pipeline", str(_ROOT / "03_rag_pipeline.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 03_rag_pipeline.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


rag = _load_rag_module()


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    max_tokens: int = Field(default=192, ge=16, le=2048)


class SourceOut(BaseModel):
    title: str
    director: str
    year: str
    genre: str
    relevance_score: float
    overview: str
    movie_id: str = ""


class PrimaryFilmOut(BaseModel):
    title: str
    director: str
    year: str
    genre: str
    movie_id: str = ""


class ChatOut(BaseModel):
    reply: str
    sources: list[SourceOut]
    primary_film: Optional[PrimaryFilmOut] = None


class TrailerOut(BaseModel):
    youtube_id: str = ""
    embed_url: str = ""
    search_url: str = ""
    source: str = ""


@dataclass
class AppState:
    vector_store: object
    model: Optional[object]
    infer_backend: str
    prompt_format: str
    chats: dict[str, object]
    lock: asyncio.Lock


state: AppState


def create_app(
    *,
    store: str,
    vector_backend: str,
    model_path: Optional[str],
    gpu_layers: int,
    prompt_format: str,
    chat_timeout: float = 90.0,
    use_generative: bool = False,
) -> FastAPI:
    global state

    vs = rag.MovieVectorStore(persist_dir=store, backend=vector_backend)
    model = None
    infer_backend = "none"
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        if p.is_dir():
            model = rag.HFMovieNerdModel(str(p))
            infer_backend = "hf"
        else:
            model = rag.MovieNerdModel(str(p), n_gpu_layers=gpu_layers)
            infer_backend = "gguf"

    state = AppState(
        vector_store=vs,
        model=model,
        infer_backend=infer_backend,
        prompt_format=prompt_format,
        chats={},
        lock=asyncio.Lock(),
    )
    infer_lock = asyncio.Lock()

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        # Pre-load the embedding model so the first chat is not stuck for minutes.
        def _warmup() -> None:
            try:
                vs.search("inception", top_k=1)
            except Exception:
                pass

        def _warm_directors() -> None:
            try:
                vs.list_director_names()
            except Exception:
                pass

        await asyncio.to_thread(_warmup)
        await asyncio.to_thread(_warm_directors)
        yield

    app = FastAPI(title="CinéBot Web", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _respond(chat_obj: object, payload: ChatIn, model: Optional[object]) -> tuple[str, list]:
        if model is None:
            return chat_obj.respond(payload.message)  # type: ignore[union-attr]

        orig = model.generate

        def _wrapped(self, prompt: str, max_tokens: int = 512, temperature: float = 0.65):
            return orig(prompt, max_tokens=payload.max_tokens, temperature=payload.temperature)

        model.generate = types.MethodType(_wrapped, model)  # type: ignore[method-assign]
        try:
            return chat_obj.respond(payload.message)  # type: ignore[union-attr]
        finally:
            model.generate = orig  # type: ignore[method-assign]

    async def _get_chat(session_id: str) -> object:
        async with state.lock:
            ch = state.chats.get(session_id)
            if ch is None:
                ch = rag.MovieNerdChat(
                    state.vector_store,
                    state.model,
                    prompt_format=state.prompt_format,
                    use_generative=use_generative,
                )  # type: ignore[arg-type]
                state.chats[session_id] = ch
            return ch

    @app.get("/")
    async def index():
        html = _UI_DIR / "index.html"
        if not html.is_file():
            raise HTTPException(status_code=500, detail=f"Missing UI file: {html}")
        # Ensure the UI is refreshed immediately (no stale cached HTML/styles).
        return FileResponse(
            html,
            headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
        )

    @app.post("/api/chat", response_model=ChatOut)
    async def chat(
        payload: ChatIn,
        x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    ):
        sid = x_session_id or str(uuid.uuid4())
        chat_obj = await _get_chat(sid)

        async with infer_lock:
            try:
                reply, retrieved = await asyncio.wait_for(
                    asyncio.to_thread(_respond, chat_obj, payload, state.model),
                    timeout=chat_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Reply took longer than {int(chat_timeout)}s. "
                        "Plot and 'tell me about' questions should be quick after restart — "
                        "if this was an opinion or recommendation, the LLM may be slow on CPU. "
                        "Try without --model, a GGUF with --gpu-layers, or wait for the first reply to finish before sending another."
                    ),
                ) from None

        sources = []
        for m in retrieved:
            sources.append(
                SourceOut(
                    title=str(m.get("title", "")),
                    director=str(m.get("director", "")),
                    year=str(m.get("year", "")),
                    genre=str(m.get("genre", "")),
                    relevance_score=float(m.get("relevance_score", 0.0)),
                    overview=str(m.get("overview", ""))[:800],
                    movie_id=str(m.get("movie_id", "")),
                )
            )

        primary = None
        top = rag.identify_primary_film(payload.message, retrieved)
        if top:
            primary = PrimaryFilmOut(
                title=str(top.get("title", "")),
                director=str(top.get("director", "")),
                year=str(top.get("year", "")),
                genre=str(top.get("genre", "")),
                movie_id=str(top.get("movie_id", "")),
            )

        resp = JSONResponse(
            content=ChatOut(reply=reply, sources=sources, primary_film=primary).model_dump()
        )
        resp.headers["X-Session-Id"] = sid
        return resp

    @app.post("/api/reset")
    async def reset(x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id")):
        # Do not wait on infer_lock — reset must stay instant even during a slow reply.
        async with state.lock:
            if x_session_id:
                ch = state.chats.get(x_session_id)
                if ch is not None:
                    ch.reset()
            else:
                for ch in state.chats.values():
                    ch.reset()
                state.chats.clear()
        return {"ok": True}

    @app.get("/api/trailer", response_model=TrailerOut)
    async def trailer(
        title: str,
        year: str = "",
        movie_id: str = "",
    ):
        if not title.strip():
            raise HTTPException(status_code=400, detail="title is required")
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(
                    rag.fetch_trailer_youtube,
                    title=title.strip(),
                    year=year.strip(),
                    movie_id=movie_id.strip(),
                ),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Trailer lookup timed out.") from None
        return TrailerOut(**data)

    @app.get("/api/health")
    async def health():
        return {
            "ok": True,
            "vector_backend": getattr(state.vector_store, "backend", "unknown"),
            "model_loaded": state.model is not None,
            "infer_backend": state.infer_backend,
            "generative_mode": use_generative,
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--store", default=rag.CHROMA_DIR)
    parser.add_argument("--vector-backend", choices=["auto", "chroma", "numpy"], default="auto")
    parser.add_argument("--model", default=None, help="Path to GGUF (optional)")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--prompt-format", choices=["phi3", "raw"], default="phi3")
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not automatically open the UI in your browser.",
    )
    parser.add_argument(
        "--chat-timeout",
        type=float,
        default=90.0,
        help="Max seconds per /api/chat reply before returning 504 (default: 90).",
    )
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Use the LLM for open-ended chat (slow on CPU). Default: fast retrieval-only answers.",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        store=args.store,
        vector_backend=args.vector_backend,
        model_path=args.model,
        gpu_layers=args.gpu_layers,
        prompt_format=args.prompt_format,
        chat_timeout=args.chat_timeout,
        use_generative=args.generative,
    )
    # Open the UI automatically once the server is reachable.
    if not args.no_open_browser:
        import threading
        import time
        import webbrowser
        import urllib.request

        url = f"http://{args.host}:{args.port}/"

        def _wait_and_open() -> None:
            # Poll briefly so we don't open the browser before the socket is bound.
            deadline = time.time() + 25
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(url, timeout=0.5).read(1)
                    print(f"Opening browser at {url}")
                    webbrowser.open(url)
                    return
                except Exception:
                    time.sleep(0.2)
            print(f"Browser auto-open timed out (server not reachable at {url}).")

        threading.Thread(target=_wait_and_open, daemon=True).start()

    if args.model and not args.generative:
        print(
            "Generative mode OFF — chat uses fast retrieval/TMDB answers (no 90s LLM waits). "
            "Pass --generative to enable open-ended LLM replies."
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
