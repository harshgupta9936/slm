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

## Anti-hallucination behavior

- Retrieval injects top-K rows (title, director, year, genre, overview) into every turn.
- The system prompt instructs the model to **only** assert DB-specific facts from that block and to admit gaps otherwise.
- Training examples from `00_synthetic_dataset.py` are **tied to CSV fields**, so the adapter learns the nerd tone without inventing metadata.

## Layout

- `databse.csv` — your cleaned movie table  
- `movie_data.py` — CSV normalization  
- `blueprint.txt` — architecture notes  
- `movie_vector_store/` — Chroma files and/or `np_rag/` embeddings after `--build`
