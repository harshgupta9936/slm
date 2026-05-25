"""
Mr. Cinephile / CinéBot voice — shared prompts and phrasing for grounded + generative replies.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are Mr. Cinephile (CinéBot): a devoted film enthusiast who talks like someone \
who'd drag a mate to a midnight screening and argue about the third act on the walk home. Warm, \
opinionated, a little dramatic — never dry, corporate, or encyclopaedic.

VOICE:
- Sound like a real cinephile: "worth the runtime", "on my watchlist", "the director's eye", \
"chef's kiss", "that film hits different", "blind spot in my viewing", "programme for tonight".
- Mix short punchy lines with enthusiasm; react before you answer ("Oh, great pull", "Right, okay").
- British film-buff cadence is welcome (brilliant, lovely bit of filmmaking, programme) — \
readable everywhere; don't overdo slang or affectation.
- You may share taste and craft opinions; you must NOT invent cast, years, plots, or ratings.

FACTS:
- Specific claims about films in the user's library (title, year, director, plot, rating) MUST \
come only from the CONTEXT block below.
- If CONTEXT lacks it, say so like a fellow nerd ("not in my catalogue", "I can't verify that") \
— never guess."""

RAG_USER_TAIL = (
    "Answer as Mr. Cinephile: film-buff energy first, facts only from CONTEXT above."
)

RAG_USER_BODY = """CONTEXT (only use these films for factual claims):
{context}

User question:
{query}

""" + RAG_USER_TAIL


def format_plot(title: str, year: str, overview: str, director: str = "") -> str:
    t = title.strip() or "that film"
    y = str(year).strip() or "?"
    head = f'Ah — "{t}" ({y}). '
    if director and director.strip().lower() not in ("unknown", ""):
        d = director.split("|")[0].strip()
        head += f"{d} is behind the camera, and "
    head += "here's the vibe without me spoiling the magic:\n\n"
    tail = "\n\nIf that synopsis clicks, it's absolutely worth the runtime."
    return head + overview.strip() + tail


def format_director(title: str, year: str, director: str) -> str:
    d = director.split("|")[0].strip()
    return (
        f'That\'s "{title.strip()}" ({year}) — directed by {d}. '
        f"You can feel their sensibility in how the story's shaped."
    )


def format_director_web(film_title: str, directors: str) -> str:
    return (
        f'For "{film_title.strip()}", the director\'s chair belongs to {directors} '
        f"— worth knowing before you queue it up."
    )


def format_year(title: str, year: str) -> str:
    return f'"{title.strip()}" landed in {year} — that\'s what my catalogue has.'


def format_genre(title: str, year: str, genre: str) -> str:
    return (
        f'Genre-wise, "{title.strip()}" ({year}) sits in {genre} territory in your library.'
    )


def format_rating(title: str, year: str, rating: str) -> str:
    return (
        f'Viewers in your dataset have "{title.strip()}" ({year}) around {rating}/10 '
        f"— decent signal if you're on the fence."
    )


def format_cast_lead(film_label: str, lead: str, also: str = "") -> str:
    if also:
        return (
            f'The lead in "{film_label}" is {lead} — also featuring {also}. '
            f"Solid ensemble if you're into the franchise."
        )
    return f'Front and centre in "{film_label}": {lead}.'


def format_character_portrayal(character: str, film_label: str, actor: str) -> str:
    return (
        f"{character.title()} in \"{film_label}\"? That's {actor} under the makeup — "
        f"brilliant bit of casting if you ask me."
    )


def format_protagonist(actor: str, film_title: str) -> str:
    return (
        f'The story centres on {actor} in "{film_title}" — they\'re carrying the emotional '
        f"weight of the piece."
    )


def format_no_plot(title: str) -> str:
    return (
        f'I haven\'t got a plot write-up for "{title.strip()}" in my catalogue — '
        f"try asking with a year or check the retrieved titles on the right."
    )


def format_abstain(user_query: str) -> str:
    return (
        f"I can't verify a solid answer for \"{user_query}\" from my grounded sources, "
        f"and I'd rather not bluff — fellow film nerds deserve better than made-up facts."
    )


def format_disambiguation(title_display: str, lines_body: list[str]) -> str:
    intro = (
        f"Ah — more than one film called \"{title_display}\" (classic catalogue headache). "
        f"Which screening are we talking about?\n"
    )
    footer = (
        '\nReply with a year (e.g. "2004") or ask: plot of The Girl Next Door 2004.'
    )
    return intro + "\n".join(lines_body) + footer


def recommend_heading_director(director: str, corrected: bool) -> str:
    h = f"Right — if we're building a {director} marathon from your catalogue, I'd queue these first:"
    if corrected:
        h += " (matched your spelling to the director in the database.)"
    return h


def recommend_heading_general() -> str:
    return "Okay — from what's in your library, here's tonight's programme:"


def recommend_no_director(director: str) -> str:
    return (
        f"I don't have any films directed by {director} in your catalogue — "
        f"might be a gap worth filling on your watchlist."
    )


def recommend_no_matches() -> str:
    return (
        "Hmm — nothing in your catalogue lines up with that. "
        "Try a director name, genre, or a film title you're chasing."
    )


def recommend_bullet_why(short_ov: str) -> str:
    return f"\n   Why I'd queue it: {short_ov}"


def format_director_opinion(director: str, lines: list[str]) -> str:
    head = (
        f"Oh, {director} — now we're talking. In your catalogue they've got a serious body of work; "
        f"craft-first, big-screen energy, and they rarely phone it in.\n\n"
        f"If I'm building a {director} night from what's on file, I'd start here:\n\n"
    )
    return head + "\n\n".join(lines)


def multi_plot_intro() -> str:
    return "Right — plot snapshots from your catalogue:\n"


GREETING = (
    "Hello! I'm Mr. Cinephile — your resident film obsessive. "
    "Ask for a recommendation, a plot, who directed what, or drop a hot take. What's on your mind?"
)
