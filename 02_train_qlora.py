"""
STEP 2 — QLoRA fine-tuning (Phi-3-mini or Mistral-7B)
Default backend: Hugging Face + PEFT + bitsandbytes (Windows-friendly).
Use --backend unsloth on Linux/WSL if unsloth is installed.

Usage:
  python 02_train_qlora.py --dataset dataset.jsonl --output ./movie-nerd-lora
  python 02_train_qlora.py --dataset dataset.jsonl --output ./movie-nerd-lora --merge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows + TRL: bundled .jinja files are UTF-8; pathlib defaults to cp1252 → UnicodeDecodeError.
if sys.platform == "win32":
    _path_read_text = Path.read_text

    def _read_text_utf8(self: Path, *args, **kwargs):
        if "encoding" not in kwargs and not args:
            kwargs["encoding"] = "utf-8"
        return _path_read_text(self, *args, **kwargs)

    Path.read_text = _read_text_utf8  # type: ignore[method-assign, assignment]

import torch
from datasets import Dataset

def hf_model_config(model_name: str):
    """
    microsoft/Phi-3-mini-4k-instruct ships rope_scaling=null; Transformers may inject a dict.
    Cached remote modeling_phi3._init_rope only accepts None or type=='longrope' — anything
    else raises (KeyError or ValueError). Normalize to None unless longrope.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(config, "model_type", None) != "phi3":
        return config
    rs = getattr(config, "rope_scaling", None)
    if not isinstance(rs, dict):
        return config
    kind = rs.get("type") or rs.get("rope_type")
    if kind == "longrope":
        config.rope_scaling = {**rs, "type": "longrope"}
    else:
        config.rope_scaling = None
    return config


MODEL_OPTIONS = {
    "phi3-mini": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "max_seq_length": 2048,
        "lora_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "vram_note": "~2.2 GB at 4-bit",
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_seq_length": 1024,
        "lora_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "vram_note": "~4+ GB at 4-bit",
    },
}


def load_dataset_from_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "instruction" not in rec or "output" not in rec:
                    print(f"  Line {i + 1}: missing instruction/output, skip")
                    continue
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"  Line {i + 1}: JSON error {e}")
    print(f"  Loaded {len(records)} examples")
    return Dataset.from_list(records)


def format_as_chat(example: dict, tokenizer) -> dict:
    system = example.get(
        "system",
        "You are CinéBot, a passionate movie nerd with encyclopedic cinema knowledge.",
    )
    user_msg = example["instruction"]
    if example.get("input"):
        user_msg += f"\n{example['input']}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _sft_config(**kwargs):
    from trl import SFTConfig

    if "max_seq_length" in kwargs:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    try:
        return SFTConfig(**kwargs)
    except TypeError:
        if "eval_strategy" in kwargs:
            kwargs = {**kwargs, "evaluation_strategy": kwargs.pop("eval_strategy")}
            return SFTConfig(**kwargs)
    raise TypeError("Could not construct SFTConfig")


def train_unsloth(args, cfg: dict, raw_ds: Dataset, output_dir: Path):
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    chat_tpl = "phi-3" if args.model == "phi3-mini" else "mistral"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=chat_tpl)

    formatted = raw_ds.map(
        lambda ex: format_as_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
        desc="Formatting",
    )
    split = formatted.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    training_args = _sft_config(
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        learning_rate=args.lr,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=str(output_dir),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        packing=True,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )
    trainer.train()

    adapter_path = output_dir / "lora-adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    if args.merge:
        merged = output_dir / "merged-model"
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        print(f"Merged model: {merged}")


def train_hf(args, cfg: dict, raw_ds: Dataset, output_dir: Path):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = raw_ds.map(
        lambda ex: format_as_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
        desc="Formatting",
    )
    split = formatted.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_config = hf_model_config(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        config=model_config,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["lora_modules"],
        ),
    )

    eval_steps = max(50, min(500, len(train_ds) // 10 or 50))
    training_args = _sft_config(
        dataset_text_field="text",
        max_length=min(cfg["max_seq_length"], 2048),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        learning_rate=args.lr,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=str(output_dir),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=max(100, eval_steps),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({round(p.total_memory / 1024**3, 1)} GB)")

    trainer.train()

    adapter_path = output_dir / "lora-adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    if args.merge:
        merge_adapter(args, cfg, adapter_path, output_dir, tokenizer)


def merge_adapter(args, cfg: dict, adapter_path: Path, output_dir: Path, tokenizer=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = output_dir / "merged-model"
    merged.mkdir(parents=True, exist_ok=True)
    offload_dir = output_dir / "merge-offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)

    merge_config = hf_model_config(cfg["model_name"])
    merge_device = args.merge_device
    if merge_device == "auto":
        merge_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Merging adapter from: {adapter_path}")
    print(f"  Merge device: {merge_device}")
    if merge_device == "cpu":
        print("  Loading base model on CPU to avoid 4 GB GPU offload errors during merge.")
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            config=merge_config,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        base = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            config=merge_config,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        base = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            offload_folder=str(offload_dir),
        )

    merged_model = base.merge_and_unload()

    # Phi-3 remote code can attach `_tied_weights_keys` as a list on some modules, but
    # Transformers' save_pretrained expects a dict-like `.keys()` here.
    for _, submodule in merged_model.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if tied is not None and not isinstance(tied, dict):
            setattr(submodule, "_tied_weights_keys", {})

    merged_model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer.save_pretrained(str(merged))
    print(f"Merged FP16 weights: {merged}")
    print(f"Temporary offload dir: {offload_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--output", default="./movie-nerd-lora")
    parser.add_argument("--model", default="phi3-mini", choices=list(MODEL_OPTIONS.keys()))
    parser.add_argument("--backend", default="hf", choices=["hf", "unsloth"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip training and merge an existing adapter saved under --output/lora-adapter.",
    )
    parser.add_argument(
        "--merge-device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for LoRA merge. Default cpu avoids offload issues on small GPUs.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Cap training rows (random subset, seed=42) to finish sooner.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Token truncate length for SFT (default: model preset). Lower=faster steps, e.g. 1024.",
    )
    args = parser.parse_args()

    cfg = {**MODEL_OPTIONS[args.model]}
    if args.max_length is not None:
        cfg["max_seq_length"] = args.max_length
    print(f"Model: {cfg['model_name']} | backend={args.backend} | {cfg['vram_note']}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        adapter_path = out / "lora-adapter"
        if not adapter_path.exists():
            raise SystemExit(f"Adapter not found: {adapter_path}")
        merge_adapter(args, cfg, adapter_path, out)
        print(f"Adapter remains under {adapter_path}")
        print(f"Merged model saved under {out / 'merged-model'}")
        return

    raw = load_dataset_from_jsonl(args.dataset)
    if len(raw) == 0:
        raise SystemExit("No training rows — run 00_synthetic_dataset.py or 01_generate_dataset.py first.")
    if args.max_train_examples is not None and len(raw) > args.max_train_examples:
        raw = raw.shuffle(seed=42).select(range(args.max_train_examples))
        print(f"  Using {len(raw)} examples (--max-train-examples)")

    if args.backend == "unsloth":
        train_unsloth(args, cfg, raw, out)
    else:
        train_hf(args, cfg, raw, out)

    print(f"Adapter saved under {out / 'lora-adapter'}")
    print("Export GGUF with llama.cpp, then: python 03_rag_pipeline.py --chat --model your.gguf")


if __name__ == "__main__":
    main()
