"""
STEP 1 — Fine-tuning dataset generator (Claude API)
Reads your movie CSV and generates conversational training examples.

Usage:
  set ANTHROPIC_API_KEY=...
  python 01_generate_dataset.py --input databse.csv --output dataset.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path

import anthropic
from tqdm import tqdm

from cinephile_voice import SYSTEM_PROMPT
from movie_data import load_movies_dataframe


def build_generation_prompt(movies_batch: list[dict]) -> str:
    movies_text = "\n".join(
        f"- Title: {m['title']} | Director: {m.get('director', 'Unknown')} | "
        f"Overview: {str(m.get('overview', 'N/A'))[:200]}"
        for m in movies_batch
    )

    return f"""Given these movies from our database, generate 3 diverse, high-quality training \
conversation examples for a movie-nerd chatbot. Cover different intents: recommendation, \
discussion, fact-finding, or comparison.

MOVIES:
{movies_text}

Return ONLY a JSON array (no markdown):
[
  {{
    "instruction": "<user message — varied phrasing, casual/fuzzy language ok>",
    "input": "",
    "output": "<Mr. Cinephile's response — warm film-buff voice, British cadence ok, never dry>"
  }}
]

Rules:
- Vary user phrasing; occasional typos/informal grammar ok
- Replies: 2–5 sentences, human, not bullet lists
- One recommendation-ish, one discussion, one fact/compare
- Use ONLY facts from the movie rows above — do not invent titles or directors"""


PERSONA_EXAMPLES = [
    {
        "instruction": "whats a good movie to watch when ur sad",
        "input": "",
        "output": "Oh man, depends on what kind of sad we're talking. If you want to cry it all out, Grave of the Fireflies will destroy you in the best way. But honestly? My go-to comfort watch is Paddington 2 — it's genuinely one of the most wholesome films ever made and you'll feel better in 20 minutes. What vibe are you going for?",
    },
    {
        "instruction": "is nolan overrated",
        "input": "",
        "output": "Okay, controversial take incoming — yes and no. His technical craft is genuinely unmatched; the practical effects in Dunkirk and the sound design in Oppenheimer are masterclasses. But his emotional storytelling can be cold, and some of his twist storytelling prioritizes cleverness over feeling. He's not overrated as a craftsman, but maybe overrated as a humanist director? The Prestige is still a near-perfect film though.",
    },
    {
        "instruction": "recommend me something like blade runner",
        "input": "",
        "output": "Oooh, chasing that neo-noir, existential sci-fi high? Ghost in the Shell (1995) is the obvious move — it asks the same questions about identity and consciousness but through anime, and it's gorgeous. Dark City from 1998 is massively underrated and hits similar vibes. If you want something more recent, Under the Skin with Scarlett Johansson is strange and haunting in exactly the right way.",
    },
]


def generate_examples_for_batch(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    prompt = build_generation_prompt(batch)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        examples = json.loads(raw)
        for ex in examples:
            ex["system"] = SYSTEM_PROMPT
        return examples
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Batch failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="databse.csv")
    parser.add_argument("--output", default="dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    df = load_movies_dataframe(args.input)
    movies = df.to_dict("records")
    if args.limit:
        movies = movies[: args.limit]
    random.shuffle(movies)

    print(f"Loaded {len(movies)} movies. Generating training data...")

    client = anthropic.Anthropic()
    all_examples: list[dict] = []
    all_examples.extend({**ex, "system": SYSTEM_PROMPT} for ex in PERSONA_EXAMPLES)
    print(f"  Added {len(PERSONA_EXAMPLES)} handcrafted persona examples")

    batches = [movies[i : i + args.batch_size] for i in range(0, len(movies), args.batch_size)]

    with tqdm(batches, desc="Generating") as pbar:
        for batch in pbar:
            all_examples.extend(generate_examples_for_batch(client, batch))
            pbar.set_postfix({"total_examples": len(all_examples)})
            time.sleep(args.delay)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
