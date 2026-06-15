from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender


REQUIRED_COLUMNS = [
    "game_id",
    "title",
    "genres",
    "tags",
    "description",
    "release_year",
    "positive_ratio",
]
OPTIONAL_COLUMNS = ["price", "developer", "publisher", "url_store", "url_ref"]

SEARCH_EXAMPLES = [
    "Sou fa de RPGs taticos e desafiadores",
    "Quero um roguelike rapido com progressao constante",
    "Procuro jogo relaxante de simulacao e crafting",
    "Gosto de fantasia sombria com historia forte",
    "Quero puzzle cooperativo com humor",
]
RECOMMENDER_CACHE_VERSION = 2
DATA_CACHE_VERSION = 2
GAMES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ludex-bg: #1b2838;
            --ludex-bg-soft: #22364a;
            --ludex-card: #16202d;
            --ludex-card-2: #101923;
            --ludex-accent: #66c0f4;
            --ludex-accent-2: #2a475e;
            --ludex-text: #dbe7f3;
            --ludex-muted: #8fa7bd;
            --ludex-good: #a4d007;
            --ludex-border: rgba(102, 192, 244, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(102, 192, 244, 0.12), transparent 32rem),
                linear-gradient(180deg, #1b2838 0%, #111923 100%);
            color: var(--ludex-text);
        }

        header[data-testid="stHeader"] {
            background: rgba(27, 40, 56, 0.88);
            border-bottom: 1px solid rgba(102, 192, 244, 0.12);
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111923 0%, #16202d 100%);
            border-right: 1px solid rgba(102, 192, 244, 0.16);
        }

        section[data-testid="stSidebar"] * {
            color: var(--ludex-text);
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #0f1822;
            border: 1px solid rgba(102, 192, 244, 0.2);
            color: var(--ludex-text);
            border-radius: 6px;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        h1, h2, h3 {
            color: #ffffff;
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(42, 71, 94, 0.82), rgba(22, 32, 45, 0.92));
            border: 1px solid rgba(102, 192, 244, 0.16);
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--ludex-muted);
            font-size: 0.78rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ludex-accent);
            font-size: 1.28rem;
        }

        .ludex-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(102, 192, 244, 0.2);
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(90deg, rgba(22, 32, 45, 0.98) 0%, rgba(34, 54, 74, 0.94) 55%, rgba(27, 40, 56, 0.82) 100%),
                radial-gradient(circle at 92% 18%, rgba(102, 192, 244, 0.28), transparent 16rem);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
        }

        .ludex-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, transparent 0%, transparent 58%, rgba(102, 192, 244, 0.08) 58%, transparent 74%);
            pointer-events: none;
        }

        .ludex-kicker {
            color: var(--ludex-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .ludex-title {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1;
            font-weight: 800;
            margin: 0;
        }

        .ludex-subtitle {
            max-width: 820px;
            color: #b8cad9;
            margin: 0.7rem 0 0;
            font-size: 1rem;
        }

        .ludex-grid-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.4rem 0 0.8rem;
        }

        .ludex-grid-title h2 {
            margin: 0;
            font-size: 1.35rem;
        }

        .ludex-active-query {
            color: var(--ludex-muted);
            margin: 0 0 0.9rem;
            font-size: 0.9rem;
        }

        .ludex-card {
            min-height: 430px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 0.72rem;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--ludex-border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(180deg, rgba(42, 71, 94, 0.84) 0%, rgba(22, 32, 45, 0.98) 34%, rgba(16, 25, 35, 0.98) 100%);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }

        .ludex-card:hover {
            transform: translateY(-2px);
            border-color: rgba(102, 192, 244, 0.42);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.38);
        }

        .ludex-card-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
        }

        .ludex-game-title {
            color: #ffffff;
            font-size: 1.08rem;
            line-height: 1.22;
            font-weight: 800;
            margin: 0;
        }

        .ludex-review {
            flex: 0 0 auto;
            color: #0b141c;
            background: var(--ludex-good);
            border-radius: 6px;
            padding: 0.25rem 0.45rem;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .ludex-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            color: var(--ludex-muted);
            font-size: 0.8rem;
        }

        .ludex-pill,
        .ludex-badge {
            display: inline-flex;
            align-items: center;
            max-width: 100%;
            border-radius: 999px;
            white-space: nowrap;
        }

        .ludex-pill {
            color: #c7d5e0;
            background: rgba(102, 192, 244, 0.1);
            border: 1px solid rgba(102, 192, 244, 0.18);
            padding: 0.25rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 650;
        }

        .ludex-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .ludex-badge {
            color: var(--ludex-accent);
            background: rgba(102, 192, 244, 0.12);
            border: 1px solid rgba(102, 192, 244, 0.2);
            padding: 0.24rem 0.52rem;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .ludex-description {
            color: #c7d5e0;
            line-height: 1.42;
            font-size: 0.9rem;
            margin: 0;
        }

        .ludex-why {
            margin-top: auto;
            background: rgba(12, 20, 28, 0.5);
            border: 1px solid rgba(102, 192, 244, 0.12);
            border-radius: 8px;
            padding: 0.65rem;
        }

        .ludex-why-title {
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0 0 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .ludex-why ul {
            margin: 0;
            padding-left: 1rem;
            color: #b8cad9;
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .ludex-score-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
        }

        .ludex-score-box {
            background: rgba(12, 20, 28, 0.55);
            border: 1px solid rgba(102, 192, 244, 0.13);
            border-radius: 6px;
            padding: 0.5rem;
        }

        .ludex-score-label {
            color: var(--ludex-muted);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .ludex-score-value {
            color: var(--ludex-accent);
            font-size: 1rem;
            font-weight: 850;
            margin-top: 0.1rem;
        }

        .ludex-actions {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
        }

        .ludex-action-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            min-height: 2rem;
            color: #ffffff !important;
            text-decoration: none !important;
            background: linear-gradient(90deg, #2a475e, #66c0f4);
            border: 1px solid rgba(102, 192, 244, 0.28);
            border-radius: 6px;
            padding: 0.4rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 850;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.2);
        }

        .ludex-action-link.secondary {
            color: #c7d5e0 !important;
            background: rgba(15, 24, 34, 0.8);
        }

        .ludex-action-link:hover {
            filter: brightness(1.08);
            color: #ffffff !important;
        }

        .ludex-empty {
            border: 1px solid rgba(102, 192, 244, 0.18);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(22, 32, 45, 0.88);
            color: #c7d5e0;
        }

        .stButton > button,
        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(90deg, #2a475e, #66c0f4);
            color: #ffffff;
            border: 0;
            border-radius: 6px;
            font-weight: 800;
        }

        .stAlert {
            background: rgba(42, 71, 94, 0.82);
            border: 1px solid rgba(102, 192, 244, 0.18);
            color: #c7d5e0;
        }

        @media (max-width: 900px) {
            .ludex-card {
                min-height: auto;
            }
            .ludex-title {
                font-size: 2.1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(total_games: int) -> None:
    total_label = f"{total_games:,}".replace(",", ".")
    st.markdown(
        f"""
        <section class="ludex-hero">
            <div class="ludex-kicker">Steam-style recommender | MVP NLP</div>
            <h1 class="ludex-title">Ludex</h1>
            <p class="ludex-subtitle">
                Descubra jogos com recomendacao hibrida: similaridade TF-IDF, busca textual,
                jogo de referencia e popularidade em uma vitrine com {total_label} titulos.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_mock_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "mock_001",
                "title": "Hollow Knight",
                "genres": "Action, Adventure, Indie",
                "tags": "metroidvania, difficult, atmospheric, exploration",
                "description": "Explore a vast ruined kingdom full of insects, secrets, bosses, and precise combat.",
                "release_year": 2017,
                "positive_ratio": 97,
            },
            {
                "game_id": "mock_002",
                "title": "Hades",
                "genres": "Action, RPG, Indie",
                "tags": "roguelike, mythology, fast-paced, story rich",
                "description": "Battle out of the underworld in a roguelike action game with sharp combat and strong characters.",
                "release_year": 2020,
                "positive_ratio": 98,
            },
            {
                "game_id": "mock_003",
                "title": "Stardew Valley",
                "genres": "RPG, Simulation, Indie",
                "tags": "farming, relaxing, crafting, life sim",
                "description": "Build a farm, make friends, explore caves, fish, craft, and restore a small rural town.",
                "release_year": 2016,
                "positive_ratio": 98,
            },
            {
                "game_id": "mock_004",
                "title": "Celeste",
                "genres": "Action, Adventure, Indie",
                "tags": "platformer, difficult, precision, emotional",
                "description": "Climb a mountain through precise platforming challenges and a personal story about persistence.",
                "release_year": 2018,
                "positive_ratio": 97,
            },
            {
                "game_id": "mock_005",
                "title": "Disco Elysium",
                "genres": "RPG",
                "tags": "detective, narrative, choices matter, political",
                "description": "Solve a murder as a troubled detective in a dense narrative RPG focused on dialogue and choices.",
                "release_year": 2019,
                "positive_ratio": 94,
            },
            {
                "game_id": "mock_006",
                "title": "Slay the Spire",
                "genres": "Strategy, Indie",
                "tags": "deckbuilding, roguelike, card game, tactical",
                "description": "Build a deck, fight tactical battles, collect relics, and climb a changing tower.",
                "release_year": 2019,
                "positive_ratio": 97,
            },
            {
                "game_id": "mock_007",
                "title": "Portal 2",
                "genres": "Action, Adventure",
                "tags": "puzzle, co-op, funny, sci-fi",
                "description": "Solve physics puzzles with portals in a sharp and funny science fiction campaign.",
                "release_year": 2011,
                "positive_ratio": 98,
            },
            {
                "game_id": "mock_008",
                "title": "The Witcher 3: Wild Hunt",
                "genres": "RPG",
                "tags": "open world, fantasy, story rich, choices matter",
                "description": "Hunt monsters and make difficult choices across a large fantasy open world.",
                "release_year": 2015,
                "positive_ratio": 96,
            },
        ]
    )


@st.cache_data(show_spinner=False)
def load_games(
    data_version: int = DATA_CACHE_VERSION,
    csv_mtime_ns: int | None = None,
) -> pd.DataFrame:
    try:
        from src.data.load_data import load_games as load_real_games
    except (ImportError, ModuleNotFoundError):
        return build_mock_games()

    games = load_real_games()
    if not isinstance(games, pd.DataFrame):
        raise TypeError("src.data.load_data.load_games() deve retornar um pandas.DataFrame.")
    return games


def games_csv_mtime_ns() -> int:
    return GAMES_CSV_PATH.stat().st_mtime_ns if GAMES_CSV_PATH.exists() else 0


@st.cache_resource(show_spinner=False)
def build_recommenders(
    games: pd.DataFrame,
    cache_version: int,
) -> tuple[ContentBasedRecommender, HybridRecommender]:
    content_recommender = ContentBasedRecommender().fit(games)
    return content_recommender, HybridRecommender(content_recommender=content_recommender)


def validate_contract(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Contrato de dados incompleto. Colunas ausentes: {missing}")

    columns = REQUIRED_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in df.columns]
    games = df[columns].copy()
    for column in ["game_id", "title", "genres", "tags", "description"]:
        games[column] = games[column].fillna("").astype(str)

    games["release_year"] = pd.to_numeric(games["release_year"], errors="coerce").fillna(0).astype(int)
    games["positive_ratio"] = pd.to_numeric(games["positive_ratio"], errors="coerce").fillna(0).clip(0, 100)
    if "price" in games.columns:
        games["price"] = pd.to_numeric(games["price"], errors="coerce").fillna(0)
    for column in ["url_store", "url_ref", "developer", "publisher"]:
        if column in games.columns:
            games[column] = games[column].fillna("").astype(str)
    return games.reset_index(drop=True)


def split_terms(values: Iterable[str]) -> list[str]:
    terms: set[str] = set()
    for value in values:
        for term in re.split(r"[,|;/]", str(value)):
            normalized = term.strip()
            if normalized:
                terms.add(normalized)
    return sorted(terms, key=str.lower)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def resolve_reference_options(games: pd.DataFrame) -> tuple[list[str | None], dict[str, str]]:
    labels_by_id = {None: "Nenhum"}
    sorted_games = games.sort_values(["title", "release_year", "game_id"])
    for _, row in sorted_games.iterrows():
        year = int(row["release_year"])
        suffix = f" ({year})" if year > 0 else ""
        labels_by_id[str(row["game_id"])] = f"{clean_display_text(row['title'])}{suffix}"
    return list(labels_by_id.keys()), labels_by_id


def filter_games(
    games: pd.DataFrame,
    recommender: HybridRecommender,
    query: str,
    reference_game_id: str | None,
    selected_genres: list[str],
    year_range: tuple[int, int],
    min_positive_ratio: int,
) -> pd.DataFrame:
    filtered = games.copy()

    filtered = filtered[filtered["positive_ratio"] >= min_positive_ratio]
    filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]

    if selected_genres:
        selected = {genre.lower() for genre in selected_genres}
        filtered = filtered[
            filtered["genres"].apply(
                lambda value: bool(selected.intersection({term.lower() for term in split_terms([value])}))
            )
        ]

    scored = recommender.score(games=games, query=query, reference_game_id=reference_game_id)
    filtered = scored.loc[filtered.index].copy()
    return filtered.sort_values(["score", "positive_ratio", "release_year"], ascending=False)


def term_set(value: str) -> set[str]:
    return {normalize_text(term) for term in split_terms([value]) if normalize_text(term)}


def query_term_matches(row: pd.Series, query: str) -> list[str]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    matches = []
    for term in split_terms([f"{row['genres']}, {row['tags']}"]):
        normalized_term = normalize_text(term)
        if normalized_term and normalized_term in normalized_query:
            matches.append(term)
    return matches[:6]


def reference_term_matches(row: pd.Series, reference_row: pd.Series | None) -> list[str]:
    if reference_row is None:
        return []

    row_terms = term_set(f"{row['genres']}, {row['tags']}")
    reference_terms = term_set(f"{reference_row['genres']}, {reference_row['tags']}")
    common = row_terms.intersection(reference_terms)
    labels = split_terms([f"{row['genres']}, {row['tags']}"])
    return [label for label in labels if normalize_text(label) in common][:6]


def build_explanation(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_row: pd.Series | None,
) -> list[str]:
    reasons = []
    query_matches = query_term_matches(row, query)
    reference_matches = reference_term_matches(row, reference_row)

    if query_matches:
        reasons.append(f"Termos da busca encontrados: {', '.join(query_matches)}.")
    elif normalize_text(query):
        reasons.append("A descricao, generos e tags ficaram proximos da sua busca pelo TF-IDF.")

    if reference_matches:
        reasons.append(f"Em comum com {reference_label}: {', '.join(reference_matches)}.")
    elif reference_row is not None:
        reasons.append(f"Similaridade textual com {reference_label} nas tags, generos e descricao.")

    if not reasons:
        reasons.append("Sem busca ou jogo de referencia, usamos popularidade como fallback.")

    reasons.append(f"Avaliacao positiva usada como estabilidade: {row['positive_ratio']:.0f}%.")
    return reasons


def clean_display_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(value: str, max_chars: int = 190) -> str:
    text = clean_display_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def safe_html(value: str) -> str:
    return html.escape(clean_display_text(value), quote=True)


def format_price(row: pd.Series) -> str:
    if "price" not in row or pd.isna(row["price"]):
        return "N/D"
    price = float(row["price"])
    if price <= 0:
        return "Free"
    return f"${price:.2f}"


def safe_url(value: str) -> str:
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return html.escape(candidate, quote=True)


def action_links_html(row: pd.Series) -> str:
    store_url = safe_url(row.get("url_store", ""))
    ref_url = safe_url(row.get("url_ref", ""))
    links = []

    if store_url:
        links.append(
            f'<a class="ludex-action-link" href="{store_url}" target="_blank" rel="noopener noreferrer">'
            "&#127918; Steam</a>"
        )
    if ref_url:
        css_class = "ludex-action-link secondary" if store_url else "ludex-action-link"
        links.append(
            f'<a class="{css_class}" href="{ref_url}" target="_blank" rel="noopener noreferrer">'
            "&#128736; PCGamingWiki</a>"
        )

    if not links:
        return '<span class="ludex-pill">Sem link externo</span>'
    return "".join(links)


def badges_html(values: str, limit: int = 4, css_class: str = "ludex-badge") -> str:
    badges = []
    for value in split_terms([values])[:limit]:
        badges.append(f'<span class="{css_class}">{safe_html(value)}</span>')
    return "".join(badges)


def reasons_html(reasons: list[str]) -> str:
    items = "".join(f"<li>{safe_html(reason)}</li>" for reason in reasons[:3])
    return f"<ul>{items}</ul>"


def render_game_card(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_row: pd.Series | None,
) -> None:
    year = int(row["release_year"])
    year_label = str(year) if year > 0 else "N/D"
    nlp_score = max(float(row["content_score"]), float(row["text_search_score"]))
    description = safe_html(truncate_text(row["description"], max_chars=170))
    genre_badges = badges_html(row["genres"], limit=4, css_class="ludex-badge")
    tag_badges = badges_html(row["tags"], limit=4, css_class="ludex-pill")
    reasons = build_explanation(row, query, reference_label, reference_row)

    st.markdown(
        f"""
        <article class="ludex-card">
            <div class="ludex-card-top">
                <h3 class="ludex-game-title">{safe_html(row["title"])}</h3>
                <div class="ludex-review">&#128077; {float(row["positive_ratio"]):.0f}%</div>
            </div>
            <div class="ludex-meta">
                <span>&#128197; {year_label}</span>
                <span>&#128176; {safe_html(format_price(row))}</span>
                <span>&#9733; {float(row["score"]):.2f}</span>
            </div>
            <div class="ludex-badges">{genre_badges}</div>
            <p class="ludex-description">{description}</p>
            <div class="ludex-badges">{tag_badges}</div>
            <div class="ludex-why">
                <div class="ludex-why-title">Por que recomendamos?</div>
                {reasons_html(reasons)}
            </div>
            <div class="ludex-score-row">
                <div class="ludex-score-box">
                    <div class="ludex-score-label">Final</div>
                    <div class="ludex-score-value">{float(row["score"]):.2f}</div>
                </div>
                <div class="ludex-score-box">
                    <div class="ludex-score-label">NLP</div>
                    <div class="ludex-score-value">{nlp_score:.2f}</div>
                </div>
                <div class="ludex-score-box">
                    <div class="ludex-score-label">Reviews</div>
                    <div class="ludex-score-value">{float(row["positive_ratio"]):.0f}%</div>
                </div>
            </div>
            <div class="ludex-actions">{action_links_html(row)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_grid(
    recommendations: pd.DataFrame,
    query: str,
    reference_label: str,
    reference_row: pd.Series | None,
) -> None:
    for start in range(0, len(recommendations), 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, recommendations.iloc[start : start + 3].iterrows()):
            with column:
                render_game_card(row, query, reference_label, reference_row)


def main() -> None:
    st.set_page_config(page_title="Ludex", page_icon="L", layout="wide")
    inject_custom_css()

    try:
        games = validate_contract(load_games(DATA_CACHE_VERSION, games_csv_mtime_ns()))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_hero(len(games))

    with st.spinner("Montando indice TF-IDF..."):
        _, hybrid_recommender = build_recommenders(games, RECOMMENDER_CACHE_VERSION)

    available_genres = split_terms(games["genres"])
    reference_options, reference_labels = resolve_reference_options(games)
    valid_years = games.loc[games["release_year"] > 0, "release_year"]
    min_year = int(valid_years.min()) if not valid_years.empty else 1980
    max_year = int(games["release_year"].max() or 2026)

    with st.sidebar:
        st.header("Ludex")
        reference_game_id = st.selectbox(
            "Jogo de referencia",
            options=reference_options,
            format_func=lambda game_id: reference_labels[game_id],
        )
        reference_label = reference_labels[reference_game_id]
        selected_example = st.selectbox("Exemplos de busca", options=[""] + SEARCH_EXAMPLES)
        custom_query = st.text_input("Preferencia, tema ou jogo", placeholder="Ex.: roguelike dificil com boa historia")
        query = custom_query.strip() or selected_example

        with st.expander("Filtros avancados", expanded=False):
            selected_genres = st.multiselect("Generos", options=available_genres)
            if min_year < max_year:
                year_range = st.slider("Ano de lancamento", min_year, max_year, (min_year, max_year))
            else:
                year_range = (min_year, max_year)
                st.caption(f"Ano de lancamento: {min_year}")
            min_positive_ratio = st.slider("Reviews positivas minimas", 0, 100, 70)
            top_n = st.slider("Quantidade de cards", 3, 24, 9)

        st.divider()
        st.caption("Fonte")
        if games["game_id"].astype(str).str.startswith("mock_").all():
            st.info("Usando dados mock ate o loader real ficar pronto.")
        else:
            st.success("Usando dados carregados por src.data.load_data.load_games().")

    recommendations = filter_games(
        games=games,
        recommender=hybrid_recommender,
        query=query,
        reference_game_id=reference_game_id,
        selected_genres=selected_genres,
        year_range=year_range,
        min_positive_ratio=min_positive_ratio,
    ).head(top_n)

    reference_row = None
    if reference_game_id:
        reference_matches = games[games["game_id"].astype(str) == str(reference_game_id)]
        if not reference_matches.empty:
            reference_row = reference_matches.iloc[0]

    left, right = st.columns([0.68, 0.32])

    with left:
        st.markdown(
            """
            <div class="ludex-grid-title">
                <h2>Top recomendacoes</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if query:
            st.markdown(f'<p class="ludex-active-query">Busca ativa: {safe_html(query)}</p>', unsafe_allow_html=True)
        if reference_label != "Nenhum":
            st.markdown(
                f'<p class="ludex-active-query">Referencia ativa: {safe_html(reference_label)}</p>',
                unsafe_allow_html=True,
            )
        if recommendations.empty:
            st.markdown(
                """
                <div class="ludex-empty">
                    Nenhum jogo encontrado com os filtros atuais. Reduza ano, genero ou reviews positivas.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_recommendation_grid(recommendations, query, reference_label, reference_row)

    with right:
        st.header("Painel")
        st.metric("Jogos carregados", len(games))
        st.metric("Jogos exibidos", len(recommendations))
        st.metric("Maior score", f"{recommendations['score'].max():.2f}" if not recommendations.empty else "0.00")

        st.subheader("Formula hibrida")
        st.caption("Com texto e referencia: 0.5 referencia + 0.3 busca textual + 0.2 popularidade.")
        st.caption("Se faltar texto ou referencia, os pesos semanticos ativos sao normalizados.")

        st.subheader("Contrato de dados")
        st.dataframe(pd.DataFrame({"coluna": REQUIRED_COLUMNS}), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
