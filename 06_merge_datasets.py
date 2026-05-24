"""
Merge base movie JSONL + intent chat add-on for full LoRA training.

Usage:
  python 06_merge_datasets.py --base dataset.jsonl --addon data/intent_chat_addon.jsonl --output dataset_full.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset.jsonl")
    ap.add_argument("--addon", default="data/intent_chat_addon.jsonl")
    ap.add_argument("--output", default="dataset_full.jsonl")
    args = ap.parse_args()

    base_path = Path(args.base)
    addon_path = Path(args.addon)
    out_path = Path(args.output)

    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base dataset: {base_path}")

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in (base_path, addon_path):
            if not path.is_file():
                print(f"  Skip (not found): {path}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        written += 1

    print(f"Merged {written} examples → {out_path}")


if __name__ == "__main__":
    main()
