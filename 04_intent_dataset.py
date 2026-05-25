"""
Build intent training data for CinéBot (rules + ML classifier + optional LoRA add-on).

Usage:
  python 04_intent_dataset.py --movies databse.csv
  python 04_intent_dataset.py --movies databse.csv --import-dir training_data
  python 06_merge_datasets.py --base dataset.jsonl --addon data/intent_chat_addon.jsonl --output dataset_full.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from movie_data import load_movies_dataframe

# Fine-grained labels → collapsed in 03_rag_pipeline.detect_intent()
INTENT_LABELS = (
    "factual_cast",
    "factual_director",
    "factual_plot",
    "factual_year",
    "factual_other",
    "recommend",
    "discussion",
)

FRANCHISE_TITLES = (
    "Harry Potter",
    "Star Wars",
    "The Lord of the Rings",
    "Marvel",
    "Batman",
    "James Bond",
    "Fast and Furious",
    "Mission Impossible",
    "The Matrix",
    "Toy Story",
)

TYPO_SWAPS = (
    ("the", "teh"),
    ("movie", "movei"),
    ("director", "directer"),
    ("recommend", "recomend"),
    ("happens", "happen"),
    ("who", "whos"),
    ("plot", "plaot"),
    ("plot", "plott"),
)

# Hand-authored traps (oversampled when --hard); mirrors real routing failures.
HARD_NEGATIVE_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("what is the plot of the girl next door", "factual_plot"),
    ("what is the plot of the girl next door 2004", "factual_plot"),
    ("tell me about the plot of the girl next door", "factual_plot"),
    ("tell me about the plaot of the girl next door", "factual_plot"),
    ("plot of the girl next door", "factual_plot"),
    ("story of the girl next door", "factual_plot"),
    ("synopsis for the girl next door", "factual_plot"),
    ("what happens in the girl next door", "factual_plot"),
    ("tell me about the girl next door", "factual_plot"),
    ("what is the girl next door about", "factual_plot"),
    ("plot of inception", "factual_plot"),
    ("what is the plot of inception", "factual_plot"),
    ("tell me about the movie inception", "factual_plot"),
    ("who directed inception", "factual_director"),
    ("who is the director of the dark knight", "factual_director"),
    ("who made the movie interstellar", "factual_director"),
    ("director of parasite", "factual_director"),
    ("best movies by christopher nolan", "recommend"),
    ("movies by christopher nolan", "recommend"),
    ("films directed by steven spielberg", "recommend"),
    ("top rated movies from denis villeneuve", "recommend"),
    ("recommend films by quentin tarantino", "recommend"),
    ("what did christopher nolan direct", "recommend"),
    ("movies from director christopher nolan", "recommend"),
    ("who has the lead role in harry potter movies", "factual_cast"),
    ("who plays harry in harry potter", "factual_cast"),
    ("lead actor in star wars", "factual_cast"),
    ("main actor in the matrix", "factual_cast"),
    ("who stars in batman movies", "factual_cast"),
    ("cast of the lord of the rings", "factual_cast"),
    ("who is the protagonist in the matrix", "factual_cast"),
    ("what year was inception released", "factual_year"),
    ("when did the dark knight come out", "factual_year"),
    ("release year of parasite", "factual_year"),
    ("is christopher nolan overrated", "discussion"),
    ("marvel vs dc which is better", "discussion"),
    ("hot take on modern horror", "discussion"),
)


def _maybe_typo(text: str, rng: random.Random, rate: float = 0.12) -> str:
    if rng.random() > rate:
        return text
    t = text
    for a, b in TYPO_SWAPS:
        if a in t.lower() and rng.random() < 0.35:
            t = re.sub(re.escape(a), b, t, count=1, flags=re.IGNORECASE)
    if rng.random() < 0.08 and len(t) > 8:
        i = rng.randint(1, len(t) - 2)
        t = t[:i] + t[i + 1 :]
    return t


def _templates_cast(title: str) -> list[str]:
    t = title
    return [
        f"who has the lead role in {t} movies",
        f"who plays the main character in {t}",
        f"lead actor in {t}",
        f"who stars in {t}",
        f"main actor in the {t} film",
        f"cast of {t}",
        f"who is the protagonist in {t}",
        f"who portrayed the hero in {t}",
        f"starring actors in {t}",
        f"who has the leading role in {t}",
        f"who played the lead in {t}",
    ]


def _templates_director(name: str) -> list[str]:
    return [
        f"movies by {name}",
        f"films directed by {name}",
        f"what did {name} direct",
        f"best movies from director {name}",
        f"recommend films by {name}",
        f"top rated {name} movies",
    ]


def _templates_plot(title: str) -> list[str]:
    return [
        f"what happens in {title}",
        f"plot of {title}",
        f"tell me about {title}",
        f"story of the movie {title}",
        f"synopsis for {title}",
        f"what is {title} about",
        f"what went on in {title}",
    ]


def _templates_year(title: str, year: str) -> list[str]:
    return [
        f"what year was {title} released",
        f"when did {title} come out",
        f"release year of {title}",
        f"what year is {title} ({year})",
    ]


def _templates_recommend(genre: str = "", director: str = "", title: str = "") -> list[str]:
    out = [
        "recommend a good thriller movie",
        "suggest something scary to watch tonight",
        "what should i watch this weekend",
        "find me a sci fi film",
        "movies like inception",
        "something similar to the dark knight",
        "best horror movies",
        "top rated action films",
    ]
    if genre:
        out += [
            f"recommend a {genre.lower()} movie",
            f"good {genre.lower()} films",
            f"i want a {genre.lower()} flick",
        ]
    if director:
        out += _templates_director(director)
    if title:
        out += [
            f"movies like {title}",
            f"something similar to {title}",
            f"if i liked {title} what else",
        ]
    return out


def _templates_discussion() -> list[str]:
    return [
        "is christopher nolan overrated",
        "marvel vs dc which is better",
        "why do people love citizen kane",
        "hot take on modern horror",
        "are sequels usually worse",
        "debate best batman actor",
        "what makes a good villain",
        "is CGI ruining movies",
        "unpopular opinion on the last jedi",
        "are remakes ever better than the original",
        "why is the godfather so highly rated",
        "is method acting overrated",
        "practical effects vs CGI your take",
        "best decade for cinema argue your case",
        "is the oscars still relevant",
        "streaming killed cinemas discuss",
        "subtitles vs dubbing for foreign films",
        "should directors cuts be the default",
        "is fan service ruining blockbusters",
        "what makes a movie a cult classic",
    ]


def _templates_director_question(title: str) -> list[str]:
    return [
        f"who directed {title}",
        f"who is the director of {title}",
        f"who made the movie {title}",
        f"director of {title}",
    ]


def _templates_factual_other(title: str) -> list[str]:
    return [
        f"what genre is {title}",
        f"how good is {title}",
        f"rating of {title}",
        f"who was gandalf in {title}",
    ]


def build_synthetic(
    movies_csv: str,
    *,
    max_movies: int | None,
    seed: int,
    typo_rate: float,
) -> list[dict]:
    rng = random.Random(seed)
    df = load_movies_dataframe(movies_csv)
    rows = df.to_dict("records")
    rng.shuffle(rows)
    if max_movies:
        rows = rows[:max_movies]

    samples: list[dict] = []
    directors_seen: set[str] = set()

    for row in rows:
        title = str(row["title"])
        year = str(row["year"])
        director = str(row["director"]).split("|")[0].strip()
        genre = str(row["genre"]).split(",")[0].strip()

        for text in _templates_cast(title):
            samples.append({"text": text, "label": "factual_cast"})
        for text in _templates_plot(title):
            samples.append({"text": text, "label": "factual_plot"})
        for text in _templates_year(title, year):
            samples.append({"text": text, "label": "factual_year"})
        for text in _templates_director_question(title):
            samples.append({"text": text, "label": "factual_director"})
        for text in _templates_factual_other(title):
            samples.append({"text": text, "label": "factual_other"})
        for text in _templates_recommend(genre=genre, title=title):
            samples.append({"text": text, "label": "recommend"})

        if director and director.lower() != "unknown" and director not in directors_seen:
            directors_seen.add(director)
            for text in _templates_director(director):
                samples.append({"text": text, "label": "recommend"})

    for franchise in FRANCHISE_TITLES:
        for text in _templates_cast(franchise):
            samples.append({"text": text, "label": "factual_cast"})
        for text in _templates_plot(franchise):
            samples.append({"text": text, "label": "factual_plot"})

    for text in _templates_discussion():
        samples.append({"text": text, "label": "discussion"})

    # Hard cases: must be factual_cast, not director/recommend
    cast_traps = [
        "who has the lead role in harry potter movies",
        "who plays harry in harry potter",
        "lead role in star wars films",
        "main actor in the matrix",
        "who stars in batman movies",
    ]
    for text in cast_traps:
        samples.append({"text": text, "label": "factual_cast"})

    if typo_rate > 0:
        augmented = []
        for s in samples:
            augmented.append(s)
            if rng.random() < 0.35:
                augmented.append(
                    {
                        "text": _maybe_typo(s["text"], rng, typo_rate),
                        "label": s["label"],
                    }
                )
        samples = augmented

    rng.shuffle(samples)
    return samples


def build_hard_negatives(*, repeats: int = 25) -> list[dict]:
    """Oversample routing traps so the classifier learns plot vs director vs cast."""
    out: list[dict] = []
    for text, label in HARD_NEGATIVE_EXAMPLES:
        for _ in range(repeats):
            out.append({"text": text, "label": label})
    return out


def import_external(import_dir: Path) -> list[dict]:
    """Load user-provided CSV/JSONL from training_data/."""
    out: list[dict] = []
    if not import_dir.is_dir():
        return out

    valid = set(INTENT_LABELS)

    for path in sorted(import_dir.rglob("*")):
        if path.suffix.lower() == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (
                        row.get("text")
                        or row.get("query")
                        or row.get("sentence")
                        or row.get("utterance")
                        or ""
                    ).strip()
                    label = (
                        row.get("label")
                        or row.get("intent")
                        or row.get("category")
                        or ""
                    ).strip()
                    if text and label in valid:
                        out.append({"text": text, "label": label})
        elif path.suffix.lower() in (".jsonl", ".json"):
            if path.suffix.lower() == ".json" and path.name.endswith("_full.json"):
                continue
            lines: list[str]
            if path.suffix.lower() == ".jsonl":
                lines = path.read_text(encoding="utf-8").splitlines()
            else:
                try:
                    blob = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(blob, list):
                    lines = [json.dumps(x, ensure_ascii=False) for x in blob]
                else:
                    continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("text") or row.get("query") or "").strip()
                label = str(row.get("label") or row.get("intent") or "").strip()
                if label not in valid:
                    mapped = _map_clinc_intent(label)
                    if mapped:
                        label = mapped
                if text and label in valid:
                    out.append({"text": text, "label": label})
    return out


def _map_clinc_intent(clinc_label: str) -> str | None:
    """Loose map from CLINC / SNIPS style labels → our schema."""
    c = clinc_label.lower()
    if "movie" in c or "watch" in c or "recommend" in c:
        return "recommend"
    if "who" in c or "when" in c or "what" in c:
        return "factual_other"
    return None


def build_chat_addon(labeled: list[dict], rng: random.Random, cap: int = 4000) -> list[dict]:
    """Optional LoRA rows: teaches tone + intent-aware reply shapes (not routing)."""
    system = (
        "You are Mr. Cinephile — talk like a devoted film enthusiast, not a helpdesk. "
        "For cast questions, discuss actors; never treat role phrases as director names."
    )
    rows: list[dict] = []
    for item in labeled:
        if item["label"] != "factual_cast":
            continue
        text = item["text"]
        title_m = re.search(
            r"(?:in|for|of)\s+(.+?)\s+(?:movies|films|series)\b",
            text,
            flags=re.IGNORECASE,
        )
        title = title_m.group(1).strip() if title_m else "that film"
        rows.append(
            {
                "instruction": text,
                "input": "",
                "output": (
                    f'Right — you want who carries the lead in the "{title}" films. '
                    f"I'll pull cast from my catalogue or the web, not pretend some character name "
                    f"is a director. Give me a sec to check properly."
                ),
                "system": system,
            }
        )
        if len(rows) >= cap:
            break
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build intent training JSONL")
    ap.add_argument("--movies", default="databse.csv")
    ap.add_argument("--max-movies", type=int, default=2500, help="Cap movies (0 = all rows)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--import-dir", default="training_data", help="Optional external datasets folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--typo-rate", type=float, default=0.12)
    ap.add_argument(
        "--hard",
        action="store_true",
        help="All movies, more typos, 25x hard-negative oversampling (for serious training)",
    )
    ap.add_argument(
        "--hard-repeats",
        type=int,
        default=25,
        help="Copies per hand-authored trap example when --hard",
    )
    ap.add_argument("--chat-addon", action="store_true", help="Write intent_chat_addon.jsonl for LoRA merge")
    args = ap.parse_args()

    if args.hard:
        if args.max_movies == 2500:
            args.max_movies = 0
        args.typo_rate = max(args.typo_rate, 0.2)

    max_movies = None if args.max_movies == 0 else args.max_movies

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_synthetic(
        args.movies,
        max_movies=max_movies,
        seed=args.seed,
        typo_rate=args.typo_rate,
    )
    if args.hard:
        hard = build_hard_negatives(repeats=args.hard_repeats)
        print(f"  Added {len(hard)} hard-negative examples ({args.hard_repeats}x traps)")
        samples.extend(hard)
    external = import_external(Path(args.import_dir))
    if external:
        print(f"  Imported {len(external)} external intent examples from {args.import_dir}/")
        samples.extend(external)

    # Dedupe
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for s in samples:
        key = (s["text"].lower().strip(), s["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    rng.shuffle(unique)

    labeled_path = out_dir / "intent_labeled.jsonl"
    with open(labeled_path, "w", encoding="utf-8") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in unique:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"Wrote {len(unique)} labeled examples -> {labeled_path}")
    for label in INTENT_LABELS:
        print(f"  {label}: {counts.get(label, 0)}")

    if args.chat_addon:
        addon = build_chat_addon(unique, rng)
        addon_path = out_dir / "intent_chat_addon.jsonl"
        with open(addon_path, "w", encoding="utf-8") as f:
            for row in addon:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(addon)} LoRA add-on examples -> {addon_path}")


if __name__ == "__main__":
    main()
