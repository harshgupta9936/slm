"""
Offline training JSONL from your movie CSV only (no API).
Every answer is grounded in row fields — safe for anti-hallucination fine-tunes.

Usage:
  python 00_synthetic_dataset.py --input databse.csv --output dataset.jsonl
  python 00_synthetic_dataset.py --input databse.csv --output dataset.jsonl --max-movies 2000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from movie_data import load_movies_dataframe

SYSTEM_PROMPT = """You are CinéBot, a passionate and opinionated movie nerd with encyclopedic \
knowledge of cinema. You speak like a real cinephile — casual, enthusiastic, sometimes dramatic. \
When discussing specific films from the user's catalog, stick to facts present in the \
retrieved/database fields; do not invent cast, directors, or plot beats."""

PERSONA_EXAMPLES = [
    {
        "instruction": "whats a good movie to watch when ur sad",
        "input": "",
        "output": "Oh man, depends on what kind of sad we're talking. If you want to cry it all out, Grave of the Fireflies will destroy you in the best way. But honestly? My go-to comfort watch is Paddington 2 — it's genuinely one of the most wholesome films ever made. What vibe are you going for?",
    },
    {
        "instruction": "is nolan overrated",
        "input": "",
        "output": "Okay, controversial take incoming — yes and no. His technical craft is genuinely unmatched. But his emotional storytelling can feel cold sometimes. He's not overrated as a craftsman, but maybe overrated as a humanist director? The Prestige is still a near-perfect film though.",
    },
    {
        "instruction": "can u recommend a horror film thats actually scary not just jumpscare garbage",
        "input": "",
        "output": "YES thank you for asking this properly. Look for stuff built on dread and character, not loud stingers — Hereditary, The Witch, Rosemary's Baby territory. The best horror earns fear from story, not noise.",
    },
]


def _genre_tokens(genre: str) -> list[str]:
    parts = [p.strip() for p in str(genre).replace("|", ",").split(",")]
    return [p for p in parts if p and p.lower() != "unknown"]


def build_examples_for_row(rows: list[dict], idx: int, rng: random.Random) -> list[dict]:
    m = rows[idx]
    title, overview, director, genre, year, rating = (
        m["title"],
        m["overview"],
        m["director"],
        m["genre"],
        m["year"],
        m["rating"],
    )
    ex: list[dict] = []

    ex.append(
        {
            "instruction": f"tell me about {title}",
            "input": "",
            "output": (
                f"Okay so {title} ({year}) — directed by {director}. "
                f"Genre-wise we're looking at {genre}. "
                f"Here's the gist: {overview} "
                f"Fans have it around a {rating}/10 vibe on the database I'm using. "
                f"Worth a look if that synopsis hits."
            ),
        }
    )

    ex.append(
        {
            "instruction": f"who directed {title}",
            "input": "",
            "output": (
                f"{director} directed {title}. "
                f"If you liked their rhythm here, dig through more of their filmography."
            ),
        }
    )

    ex.append(
        {
            "instruction": f"what year did {title} come out",
            "input": "",
            "output": f"{title} is listed as {year} in my movie database.",
        }
    )

    same_dir = [r for r in rows if r["director"] == director and r["title"] != title]
    if same_dir:
        other = rng.choice(same_dir)
        ov = other["overview"]
        tail = "..." if len(ov) > 280 else ""
        ex.append(
            {
                "instruction": f"i liked {title} what else by the same director",
                "input": "",
                "output": (
                    f"If {director} clicked for you with {title}, check out {other['title']} ({other['year']}) — "
                    f"{ov[:280]}{tail}"
                ),
            }
        )

    tags = _genre_tokens(genre)
    if tags:
        tag = rng.choice(tags)
        candidates = [
            r
            for r in rows
            if r["title"] != title and tag.lower() in str(r["genre"]).lower()
        ]
        if candidates:
            pick = rng.choice(candidates)
            pov = pick["overview"]
            tail = "..." if len(pov) > 240 else ""
            ex.append(
                {
                    "instruction": f"recommend something {tag.lower()} like {title}",
                    "input": "",
                    "output": (
                        f"From the same rough {tag} lane in this database, I'd nudge you toward {pick['title']} ({pick['year']}) "
                        f"by {pick['director']}. {pov[:240]}{tail}"
                    ),
                }
            )

    ov = overview
    tail = "..." if len(ov) > 220 else ""
    ex.append(
        {
            "instruction": f"is {title} worth watching",
            "input": "",
            "output": (
                f"Honestly? {title} sits at about {rating}/10 with voters in this dataset — that's a solid signal. "
                f"{ov[:220]}{tail} "
                f"If that premise sounds like your jam, yeah, queue it."
            ),
        }
    )

    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="databse.csv")
    ap.add_argument("--output", default="dataset.jsonl")
    ap.add_argument("--max-movies", type=int, default=None, help="Cap rows for a quicker experiment")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = load_movies_dataframe(args.input)
    all_rows = df.to_dict("records")
    indices = list(range(len(all_rows)))
    rng.shuffle(indices)
    if args.max_movies is not None:
        indices = indices[: args.max_movies]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in PERSONA_EXAMPLES:
            row = {**ex, "system": SYSTEM_PROMPT}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

        for idx in indices:
            for ex in build_examples_for_row(all_rows, idx, rng):
                row = {**ex, "system": SYSTEM_PROMPT}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
