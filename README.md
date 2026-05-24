# Movie Nerd SLM (CinéBot)

End-to-end stack: **grounded RAG** over your movie table + **optional QLoRA fine-tune** for a passionate cinephile voice. Factual claims about films in your CSV should come from **retrieved context**, not from bare model memory.

Your repo includes a cleaned dataset as **`databse.csv`** (columns: `movie_title`, `overview`, `director`, `genre`, `rating`, `release_year`, …). All scripts normalize this automatically via `movie_data.py`.

## Pipeline

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `00_synthetic_dataset.py` | Build `dataset.jsonl` from the CSV only (no API, fully grounded). |
| 1 | `01_generate_dataset.py` | Optional: richer examples via Anthropic API. |
| 2 | `02_train_qlora.py` | QLoRA fine-tune Phi-3-mini or Mistral (default backend: Hugging Face + PEFT, Windows-friendly). |
| 3 | `03_rag_pipeline.py` | Build vector index + chat / recommend (GGUF via `llama-cpp-python`). |

## Quick start (RAG only)

1. Create a venv and install dependencies (if `pip` errors, run `python -m ensurepip --upgrade` or repair your Python install):

```powershell
cd C:\Users\JOEY\Documents\slm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Build the index (uses **Chroma** if installed, otherwise **NumPy + MiniLM** automatically):

```powershell
python 03_rag_pipeline.py --build --movies databse.csv
```

3. Chat **without** a local GGUF (retrieval-only smoke test):

```powershell
python 03_rag_pipeline.py --chat
```

4. Chat **with** a GGUF (e.g. after you merge/export a Phi-3-mini model):

```powershell
python 03_rag_pipeline.py --chat --model .\movie-nerd.Q4_K_M.gguf
```

Force the lightweight store:

```powershell
python 03_rag_pipeline.py --build --movies databse.csv --vector-backend numpy
```

## Training data (offline)

Generate a large grounded JSONL from your CSV:

```powershell
python 00_synthetic_dataset.py --input databse.csv --output dataset.jsonl
```

For a quick experiment:

```powershell
python 00_synthetic_dataset.py --input databse.csv --output dataset_small.jsonl --max-movies 500
```

## Fine-tuning (GPU recommended)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes peft trl
python 02_train_qlora.py --dataset dataset.jsonl --output .\movie-nerd-lora --model phi3-mini --backend hf
```

On **Windows**, if you still see a `UnicodeDecodeError` from `trl` / `.jinja` files, run with UTF-8 mode: `python -X utf8 02_train_qlora.py ...` (the training script also patches `Path.read_text` to default to UTF-8 when possible).

Merge LoRA into full weights for GGUF export:

```powershell
python 02_train_qlora.py --dataset dataset.jsonl --output .\movie-nerd-lora --merge --epochs 1 --model phi3-mini
```

Then convert the merged HF folder to GGUF using [llama.cpp](https://github.com/ggerganov/llama.cpp) `convert_hf_to_gguf.py` (see upstream docs for your exact model family).

On **Linux/WSL** with Unsloth installed, you can use `--backend unsloth` for faster training.

## Intent understanding (train once)

Routing (cast vs director vs plot vs recommend) uses **rules + a small classifier** — not only the big chat model.

```powershell
pip install scikit-learn joblib
python 04_intent_dataset.py --movies databse.csv --chat-addon
python train_intent_classifier.py
```

Optional external data: see [training_data/EXTERNAL_DATASETS.md](training_data/EXTERNAL_DATASETS.md) (CLINC150 link included).

To also fine-tune the **chat** model with intent-aware examples:

```powershell
python 00_synthetic_dataset.py --input databse.csv --output dataset.jsonl
python 06_merge_datasets.py --base dataset.jsonl --addon data/intent_chat_addon.jsonl --output dataset_full.jsonl
python 02_train_qlora.py --dataset dataset_full.jsonl --output .\movie-nerd-lora --merge
```

Restart `web_chat.py` after training the classifier.

## TMDB API key (persistent)

Trailers and web plot fallback use [TMDB](https://www.themoviedb.org/). Save your key once — it survives reboots and restarts.

**Option A — project `.env` (recommended)**

```powershell
pip install python-dotenv
copy .env.example .env
# Edit .env and set TMDB_API_KEY=...
```

Or run the helper (prompts for your key):

```powershell
.\scripts\set_tmdb_key.ps1
```

**Option B — Windows user environment (every app on your account)**

```powershell
.\scripts\set_tmdb_key.ps1 -AlsoSetWindowsUserEnv
# Or manually:
setx TMDB_API_KEY "your_key_here"
```

After `setx`, open a **new** terminal. `.env` is loaded automatically when you run any script that imports `movie_data.py`.

## Anti-hallucination behavior

- Retrieval injects top-K rows (title, director, year, genre, overview) into every turn.
- The system prompt instructs the model to **only** assert DB-specific facts from that block and to admit gaps otherwise.
- Training examples from `00_synthetic_dataset.py` are **tied to CSV fields**, so the adapter learns the nerd tone without inventing metadata.

## Layout

- `databse.csv` — your cleaned movie table  
- `movie_data.py` — CSV normalization  
- `blueprint.txt` — architecture notes  
- `movie_vector_store/` — Chroma files and/or `np_rag/` embeddings after `--build`
