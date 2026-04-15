"""
hi 

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


class ChatOut(BaseModel):
    reply: str
    sources: list[SourceOut]


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
                ch = rag.MovieNerdChat(state.vector_store, state.model, prompt_format=state.prompt_format)  # type: ignore[arg-type]
                state.chats[session_id] = ch
            return ch

    @app.get("/")
    async def index():
        html = _UI_DIR / "index.html"
        if not html.is_file():
            raise HTTPException(status_code=500, detail=f"Missing UI file: {html}")
        return FileResponse(html)

    @app.post("/api/chat", response_model=ChatOut)
    async def chat(
        payload: ChatIn,
        x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    ):
        sid = x_session_id or str(uuid.uuid4())
        chat_obj = await _get_chat(sid)

        async with infer_lock:
            reply, retrieved = await asyncio.to_thread(_respond, chat_obj, payload, state.model)

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
                )
            )

        resp = JSONResponse(content=ChatOut(reply=reply, sources=sources).model_dump())
        resp.headers["X-Session-Id"] = sid
        return resp

    @app.post("/api/reset")
    async def reset(x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id")):
        if not x_session_id:
            raise HTTPException(status_code=400, detail="Missing X-Session-Id")
        async with infer_lock:
            async with state.lock:
                ch = state.chats.get(x_session_id)
                if ch is not None:
                    ch.reset()
        return {"ok": True}

    @app.get("/api/health")
    async def health():
        return {
            "ok": True,
            "vector_backend": getattr(state.vector_store, "backend", "unknown"),
            "model_loaded": state.model is not None,
            "infer_backend": state.infer_backend,
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
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        store=args.store,
        vector_backend=args.vector_backend,
        model_path=args.model,
        gpu_layers=args.gpu_layers,
        prompt_format=args.prompt_format,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
