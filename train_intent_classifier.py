"""
Train a lightweight intent classifier (MiniLM embeddings + logistic regression).

Usage:
  python 04_intent_dataset.py --movies databse.csv --chat-addon
  python train_intent_classifier.py --data data/intent_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from movie_data import load_project_env

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = Path(__file__).resolve().parent / "intent_model"


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


def main():
    load_project_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intent_labeled.jsonl")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python 04_intent_dataset.py --movies databse.csv"
        )

    texts, labels = load_labeled(data_path)
    if len(texts) < 100:
        raise RuntimeError(f"Need more examples (got {len(texts)}). Run 04_intent_dataset.py first.")

    enc_labels = LabelEncoder()
    y = enc_labels.fit_transform(labels)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        texts,
        y,
        texts,
        test_size=args.test_size,
        random_state=args.seed,
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

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    print("\n" + classification_report(y_test, y_pred, target_names=list(enc_labels.classes_)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "classifier.joblib")
    meta = {
        "embed_model": EMBED_MODEL,
        "labels": list(enc_labels.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved intent classifier -> {out_dir}")
    print("Restart web_chat.py — detect_intent() will use the model automatically.")


if __name__ == "__main__":
    main()
