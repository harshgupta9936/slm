"""
Train intent classifier for CinéBot routing.

Modes:
  fast  — MiniLM embeddings + logistic regression (~2–10 min CPU)
  hard  — MiniLM embeddings + 3-layer MLP (typ. 1–4 h CPU; stays under 8 h)

Usage:
  python 04_intent_dataset.py --movies databse.csv --hard
  python train_intent_classifier.py --mode hard
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from movie_data import load_project_env

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = Path(__file__).resolve().parent / "intent_model"


def stratified_cap(
    texts: list[str], labels: list[str], max_samples: int, seed: int
) -> tuple[list[str], list[str]]:
    """Downsample while keeping label balance (stays within CPU time budget)."""
    if len(texts) <= max_samples:
        return texts, labels
    import pandas as pd

    df = pd.DataFrame({"text": texts, "label": labels})
    parts = []
    per_label = max(50, max_samples // df["label"].nunique())
    for lab in df["label"].unique():
        chunk = df[df["label"] == lab]
        n = min(len(chunk), per_label)
        parts.append(resample(chunk, replace=False, n_samples=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    if len(out) > max_samples:
        out = out.iloc[:max_samples]
    print(f"  Subsampled {len(texts)} -> {len(out)} for hard training (time budget)")
    return out["text"].tolist(), out["label"].tolist()


def load_labeled(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = str(row.get("text", "")).strip()
            lab = str(row.get("label", "")).strip()
            if t and lab:
                texts.append(t)
                labels.append(lab)
    return texts, labels


def train_fast(
    texts: list[str],
    labels: list[str],
    out_dir: Path,
    *,
    test_size: float,
    seed: int,
) -> None:
    enc_labels = LabelEncoder()
    y = enc_labels.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"Encoding {len(texts)} examples with {EMBED_MODEL}...")
    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
    X_train_emb = encoder.encode(
        X_train,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_test_emb = encoder.encode(
        X_test,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(max_iter=4000, class_weight="balanced")
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    print("\n" + classification_report(y_test, y_pred, target_names=list(enc_labels.classes_)))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "classifier.joblib")
    meta = {
        "model_type": "logistic",
        "embed_model": EMBED_MODEL,
        "labels": list(enc_labels.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_hard(
    texts: list[str],
    labels: list[str],
    out_dir: Path,
    *,
    test_size: float,
    seed: int,
    epochs: int,
    batch_size: int,
    max_hours: float,
) -> None:
    """
    Hard mode: MiniLM embeddings + 3-layer MLP head (stronger than logistic).
    Uses the full dataset when subsampling is disabled; typical runtime 1–4 h CPU.
    """
    enc_labels = LabelEncoder()
    y = enc_labels.fit_transform(labels)
    label_names = list(enc_labels.classes_)

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"Hard train: {len(train_texts)} train / {len(test_texts)} test")
    print(f"Encoder: {EMBED_MODEL}  |  MLP epochs~{epochs * 15}  |  budget <= {max_hours:.1f}h")

    t0 = time.time()
    encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
    X_train = encoder.encode(
        train_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    X_test = encoder.encode(
        test_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(512, max(64, batch_size * 4)),
        learning_rate_init=1e-3,
        max_iter=max(40, epochs * 15),
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=10,
        verbose=True,
        random_state=seed,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=label_names))

    elapsed_h = (time.time() - t0) / 3600.0
    if elapsed_h > max_hours:
        print(f"Warning: training took {elapsed_h:.2f}h (budget was {max_hours}h)")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"clf": mlp, "encoder_model": EMBED_MODEL, "label_encoder": enc_labels},
        out_dir / "classifier.joblib",
    )
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

    meta = {
        "model_type": "mlp",
        "embed_model": EMBED_MODEL,
        "labels": label_names,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "mlp_layers": [512, 256, 128],
        "elapsed_hours": round(elapsed_h, 3),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    load_project_env()
    ap = argparse.ArgumentParser(description="Train CinéBot intent classifier")
    ap.add_argument("--data", default="data/intent_labeled.jsonl")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mode", choices=("fast", "hard"), default="hard")
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4, help="Hard mode only")
    ap.add_argument("--batch-size", type=int, default=32, help="Hard mode only")
    ap.add_argument(
        "--max-hours",
        type=float,
        default=7.5,
        help="Soft time budget (hard mode); reduce --epochs if you hit it often",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Hard mode: cap training rows (0 = use all) to finish under ~8h CPU",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing {data_path}. Run:\n"
            f"  python 04_intent_dataset.py --movies databse.csv --hard"
        )

    texts, labels = load_labeled(data_path)
    if len(texts) < 100:
        raise RuntimeError(f"Need more examples (got {len(texts)}). Run 04_intent_dataset.py first.")

    if args.mode == "hard" and args.max_samples > 0 and len(texts) > args.max_samples:
        texts, labels = stratified_cap(texts, labels, args.max_samples, args.seed)

    out_dir = Path(args.out)
    if out_dir.exists() and args.mode == "hard":
        shutil.rmtree(out_dir, ignore_errors=True)

    if args.mode == "fast":
        train_fast(texts, labels, out_dir, test_size=args.test_size, seed=args.seed)
    else:
        train_hard(
            texts,
            labels,
            out_dir,
            test_size=args.test_size,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_hours=args.max_hours,
        )

    print(f"\nSaved intent classifier ({args.mode}) -> {out_dir}")
    print("Restart web_chat.py — detect_intent() loads the model automatically.")


if __name__ == "__main__":
    main()
