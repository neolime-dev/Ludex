from __future__ import annotations

import base64
import html
import pickle
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

from src.recommenders.content_based import ContentBasedRecommender, RECOMMENDER_MODEL_VERSION
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.semantic import SemanticRecommender
from src.recommenders.embeddings import create_embedding_provider
from src.recommenders.ollama_explainer import OllamaExplainer


REQUIRED_COLUMNS = [
    "game_id",
    "title",
    "genres",
    "tags",
    "description",
    "release_year",
    "positive_ratio",
]
OPTIONAL_COLUMNS = [
    "price",
    "developer",
    "publisher",
    "url_store",
    "url_ref",
    "header_image",
    "steam_appid",
    "review_keywords",
    "sentiment_score",
]

RECOMMENDER_CACHE_VERSION = RECOMMENDER_MODEL_VERSION
DATA_CACHE_VERSION = 3
GAMES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"
MODEL_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "cache" / "tfidf_model.pkl"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ludex-bg: #09090b;
            --ludex-bg-soft: #111827;
            --ludex-panel: #18181b;
            --ludex-panel-2: #0f172a;
            --ludex-card: #18181b;
            --ludex-card-top: #27272a;
            --ludex-line: rgba(212, 212, 216, 0.12);
            --ludex-line-strong: rgba(212, 212, 216, 0.22);
            --ludex-text: #fafafa;
            --ludex-muted: #a1a1aa;
            --ludex-soft: #d4d4d8;
            --ludex-accent: #38bdf8;
            --ludex-accent-2: #22c55e;
            --ludex-warn: #facc15;
            --ludex-shadow: 0 24px 70px rgba(0, 0, 0, 0.42), 0 1px 0 rgba(255, 255, 255, 0.04) inset;
        }

        .ludex-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 1.55rem 1.65rem;
            margin-bottom: 1.4rem;
            background:
                linear-gradient(100deg, rgba(24, 24, 27, 0.98) 0%, rgba(15, 23, 42, 0.94) 58%, rgba(9, 9, 11, 0.96) 100%),
                linear-gradient(135deg, rgba(56, 189, 248, 0.18), transparent 42%),
                linear-gradient(215deg, rgba(34, 197, 94, 0.14), transparent 36%);
            box-shadow: var(--ludex-shadow);
        }

        .ludex-kicker {
            color: var(--ludex-soft);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .ludex-title {
            color: var(--ludex-text);
            font-size: 3.05rem;
            line-height: 1;
            font-weight: 900;
            margin: 0;
            margin-bottom: 0.75rem;
        }

        .ludex-subtitle {
            max-width: 820px;
            color: var(--ludex-muted);
            margin: 0.7rem 0 0;
            font-size: 1rem;
            line-height: 1.55;
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
            font-size: 1.15rem;
            font-weight: 850;
        }

        .ludex-active-query {
            color: var(--ludex-muted);
            margin: 0 0 0.9rem;
            font-size: 0.9rem;
        }

        .ludex-search-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 0.25rem;
            color: var(--ludex-text);
            font-size: 1rem;
            font-weight: 900;
        }

        .ludex-search-copy {
            margin: 0 0 0.85rem;
            color: var(--ludex-muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .ludex-icon {
            width: 0.95rem;
            height: 0.95rem;
            display: inline-block;
            flex: 0 0 auto;
            background: currentColor;
            vertical-align: -0.16rem;
            mask-position: center;
            mask-repeat: no-repeat;
            mask-size: contain;
            -webkit-mask-position: center;
            -webkit-mask-repeat: no-repeat;
            -webkit-mask-size: contain;
        }

        .ludex-icon-calendar {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/calendar.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/calendar.svg');
        }

        .ludex-icon-coins {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/coins.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/coins.svg');
        }

        .ludex-icon-thumbs-up {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/thumbs-up.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/thumbs-up.svg');
        }

        .ludex-icon-gamepad {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/gamepad-2.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/gamepad-2.svg');
        }

        .ludex-icon-wrench {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/wrench.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/wrench.svg');
        }

        .ludex-icon-search {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/search.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/search.svg');
        }

        .ludex-icon-palette {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/palette.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/palette.svg');
        }

        .ludex-icon-joystick {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/joystick.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/joystick.svg');
        }

        .ludex-icon-badge-dollar-sign {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/badge-dollar-sign.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/badge-dollar-sign.svg');
        }

        .ludex-icon-building-2 {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/building-2.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/building-2.svg');
        }

        .ludex-icon-database {
            mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/database.svg');
            -webkit-mask-image: url('https://unpkg.com/lucide-static@0.468.0/icons/database.svg');
        }

        .ludex-sidebar-brand {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.25rem 0 1rem;
            background:
                linear-gradient(180deg, rgba(39, 39, 42, 0.8), rgba(9, 9, 11, 0.84)),
                linear-gradient(135deg, rgba(56, 189, 248, 0.1), transparent 48%);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
        }

        .ludex-sidebar-title {
            margin: 0;
            color: var(--ludex-text);
            font-size: 1.25rem;
            font-weight: 900;
        }

        .ludex-sidebar-subtitle {
            margin: 0.3rem 0 0;
            color: var(--ludex-muted);
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .ludex-filter-heading {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 1rem 0 0.45rem;
            color: var(--ludex-soft);
            font-size: 0.75rem;
            font-weight: 900;
            text-transform: uppercase;
        }

        .ludex-sidebar-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin-top: 0.9rem;
            background: rgba(9, 9, 11, 0.42);
            color: var(--ludex-soft);
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .ludex-ai-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
            margin: 0.25rem 0 0.75rem;
        }

        .ludex-ai-card {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.62rem;
            background: rgba(9, 9, 11, 0.42);
        }

        .ludex-ai-label {
            display: block;
            color: var(--ludex-muted);
            font-size: 0.62rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .ludex-ai-value {
            display: block;
            margin-top: 0.18rem;
            color: var(--ludex-text);
            font-size: 0.96rem;
            font-weight: 900;
        }

        .ludex-ai-note {
            margin: 0;
            color: var(--ludex-soft);
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .ludex-card {
            min-height: 475px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 0.78rem;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 1.05rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(180deg, rgba(39, 39, 42, 0.98) 0%, rgba(24, 24, 27, 0.98) 42%, rgba(9, 9, 11, 0.98) 100%);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }

        .ludex-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto;
            height: 3px;
            background: linear-gradient(90deg, var(--ludex-accent), var(--ludex-accent-2));
            opacity: 0.72;
        }

        .ludex-card:hover {
            transform: translateY(-4px);
            border-color: rgba(250, 250, 250, 0.24);
            box-shadow: 0 28px 72px rgba(0, 0, 0, 0.44);
        }

        .ludex-card-media {
            position: relative;
            height: 9.25rem;
            margin: -1.05rem -1.05rem 0;
            overflow: hidden;
            border-bottom: 1px solid var(--ludex-line);
            background:
                linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(34, 197, 94, 0.1)),
                linear-gradient(180deg, rgba(39, 39, 42, 0.95), rgba(9, 9, 11, 0.95));
        }

        .ludex-card-media::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(180deg, transparent 28%, rgba(9, 9, 11, 0.2) 72%, rgba(9, 9, 11, 0.62) 100%),
                linear-gradient(90deg, rgba(9, 9, 11, 0.26), transparent 46%);
            pointer-events: none;
        }

        .ludex-card-image {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            object-position: center;
            transform: scale(1.01);
            transition: transform 240ms ease, filter 240ms ease;
        }

        .ludex-card:hover .ludex-card-image {
            transform: scale(1.055);
            filter: saturate(1.08) contrast(1.04);
        }

        .ludex-card-image-fallback {
            display: none;
            width: 100%;
            height: 100%;
            align-items: center;
            justify-content: center;
            color: rgba(250, 250, 250, 0.52);
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
        }

        .ludex-card-image-fallback.visible {
            display: flex;
        }

        .ludex-card-top {
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
        }

        .ludex-game-title {
            color: var(--ludex-text);
            font-size: 1.05rem;
            line-height: 1.22;
            font-weight: 850;
            margin: 0;
        }

        .ludex-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            color: var(--ludex-muted);
            font-size: 0.8rem;
        }

        .ludex-meta-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            border: 1px solid var(--ludex-line);
            border-radius: 999px;
            padding: 0.26rem 0.55rem;
            background: rgba(9, 9, 11, 0.48);
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
            color: var(--ludex-soft);
            background: rgba(39, 39, 42, 0.72);
            border: 1px solid var(--ludex-line);
            padding: 0.24rem 0.52rem;
            font-size: 0.7rem;
            font-weight: 700;
        }

        .ludex-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .ludex-badge {
            color: #e0f2fe;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 0.24rem 0.52rem;
            font-size: 0.7rem;
            font-weight: 750;
        }

        .ludex-description {
            color: var(--ludex-soft);
            line-height: 1.48;
            font-size: 0.88rem;
            margin: 0;
        }

        .ludex-why {
            margin-top: auto;
            background: rgba(9, 9, 11, 0.46);
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.72rem;
        }

        .ludex-community {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.1), rgba(9, 9, 11, 0.32));
            border: 1px solid rgba(34, 197, 94, 0.18);
            border-radius: 8px;
            padding: 0.72rem;
        }

        .ludex-community-title {
            color: var(--ludex-text);
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0;
            margin: 0 0 0.4rem;
            text-transform: uppercase;
        }

        .ludex-community-line {
            color: var(--ludex-soft);
            font-size: 0.78rem;
            line-height: 1.35;
            margin: 0;
        }

        .ludex-why-title {
            color: var(--ludex-text);
            font-size: 0.68rem;
            font-weight: 900;
            margin: 0 0 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .ludex-why ul {
            margin: 0;
            padding-left: 1rem;
            color: var(--ludex-soft);
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .ludex-details {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            background: rgba(9, 9, 11, 0.36);
            padding: 0.64rem 0.72rem;
        }

        .ludex-details summary {
            cursor: pointer;
            color: var(--ludex-soft);
            font-size: 0.72rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .ludex-details .ludex-community,
        .ludex-details .ludex-score-row {
            margin-top: 0.64rem;
        }

        .ludex-score-row {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.45rem;
        }

        .ludex-score-box {
            background: rgba(9, 9, 11, 0.42);
            border: 1px solid var(--ludex-line);
            border-radius: 6px;
            padding: 0.52rem;
        }

        .ludex-score-label {
            color: var(--ludex-muted);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .ludex-score-value {
            color: var(--ludex-text);
            font-size: 1rem;
            font-weight: 900;
            margin-top: 0.1rem;
        }

        .ludex-score-track {
            position: relative;
            height: 0.42rem;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(63, 63, 70, 0.66);
            border: 1px solid rgba(250, 250, 250, 0.06);
        }

        .ludex-score-fill {
            width: var(--score-width);
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--ludex-accent), var(--ludex-accent-2));
            box-shadow: 0 0 22px rgba(56, 189, 248, 0.18);
        }

        .ludex-actions {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            padding-top: 0.12rem;
        }

        .ludex-action-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            min-height: 2rem;
            color: #ffffff !important;
            text-decoration: none !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.95), rgba(34, 197, 94, 0.86));
            border: 1px solid rgba(250, 250, 250, 0.18);
            border-radius: 6px;
            padding: 0.4rem 0.7rem;
            font-size: 0.74rem;
            font-weight: 850;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
            transition: transform 160ms ease, filter 160ms ease;
        }

        .ludex-action-link.secondary {
            color: var(--ludex-soft) !important;
            background: rgba(39, 39, 42, 0.82);
        }

        .ludex-action-link:hover {
            filter: brightness(1.08);
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        .ludex-empty {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(24, 24, 27, 0.88);
            color: var(--ludex-soft);
        }

        .ludex-side-panel {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 1rem;
            background:
                linear-gradient(180deg, rgba(39, 39, 42, 0.86), rgba(9, 9, 11, 0.9)),
                linear-gradient(135deg, rgba(56, 189, 248, 0.08), transparent 42%);
            box-shadow: var(--ludex-shadow);
        }

        .ludex-panel-title {
            margin: 0 0 0.8rem;
            color: var(--ludex-text);
            font-size: 0.92rem;
            font-weight: 900;
        }

        .ludex-panel-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .ludex-panel-stat {
            min-width: 0;
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.62rem;
            background: rgba(9, 9, 11, 0.48);
        }

        .ludex-panel-label {
            display: block;
            color: var(--ludex-muted);
            font-size: 0.64rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .ludex-panel-value {
            display: block;
            margin-top: 0.24rem;
            color: var(--ludex-text);
            font-size: 1.05rem;
            font-weight: 900;
        }

        .ludex-panel-copy {
            color: var(--ludex-soft);
            font-size: 0.8rem;
            line-height: 1.45;
            margin: 0 0 0.8rem;
        }

        .ludex-contract {
            display: flex;
            flex-wrap: wrap;
            gap: 0.36rem;
        }

        @media (max-width: 900px) {
            .ludex-card {
                min-height: auto;
            }
            .ludex-score-row {
                grid-template-columns: 1fr;
            }
            .ludex-panel-grid {
                grid-template-columns: 1fr;
            }
            .ludex-ai-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_base64_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_hero(total_games: int) -> None:
    total_label = f"{total_games:,}".replace(",", ".")
    st.markdown(
        f"""
        <div class="ludex-hero">
            <div class="ludex-kicker">Ludex Recommender | Premium NLP Build</div>
            <h1 class="ludex-title">Ludex</h1>
            <p class="ludex-subtitle">
                Uma vitrine curada com {total_label} titulos, ranking hibrido por similaridade,
                intencao de busca, sinais opinativos e qualidade da comunidade.
            </p>
        </div>
        """.replace(",", "."),
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
                "review_keywords": "immersive, challenging, atmospheric, emotional",
                "sentiment_score": 0.92,
            },
            {
                "game_id": "mock_002",
                "title": "Hades",
                "genres": "Action, RPG, Indie",
                "tags": "roguelike, mythology, fast-paced, story rich",
                "description": "Battle out of the underworld in a roguelike action game with sharp combat and strong characters.",
                "release_year": 2020,
                "positive_ratio": 98,
                "review_keywords": "addictive, satisfying, fast-paced, replayable",
                "sentiment_score": 0.95,
            },
            {
                "game_id": "mock_003",
                "title": "Stardew Valley",
                "genres": "RPG, Simulation, Indie",
                "tags": "farming, relaxing, crafting, life sim",
                "description": "Build a farm, make friends, explore caves, fish, craft, and restore a small rural town.",
                "release_year": 2016,
                "positive_ratio": 98,
                "review_keywords": "relaxing, cozy, addictive, charming",
                "sentiment_score": 0.94,
            },
            {
                "game_id": "mock_004",
                "title": "Celeste",
                "genres": "Action, Adventure, Indie",
                "tags": "platformer, difficult, precision, emotional",
                "description": "Climb a mountain through precise platforming challenges and a personal story about persistence.",
                "release_year": 2018,
                "positive_ratio": 97,
                "review_keywords": "emotional, difficult, rewarding, precise",
                "sentiment_score": 0.93,
            },
            {
                "game_id": "mock_005",
                "title": "Disco Elysium",
                "genres": "RPG",
                "tags": "detective, narrative, choices matter, political",
                "description": "Solve a murder as a troubled detective in a dense narrative RPG focused on dialogue and choices.",
                "release_year": 2019,
                "positive_ratio": 94,
                "review_keywords": "deep, political, emotional, narrative",
                "sentiment_score": 0.9,
            },
            {
                "game_id": "mock_006",
                "title": "Slay the Spire",
                "genres": "Strategy, Indie",
                "tags": "deckbuilding, roguelike, card game, tactical",
                "description": "Build a deck, fight tactical battles, collect relics, and climb a changing tower.",
                "release_year": 2019,
                "positive_ratio": 97,
                "review_keywords": "addictive, strategic, replayable, challenging",
                "sentiment_score": 0.91,
            },
            {
                "game_id": "mock_007",
                "title": "Portal 2",
                "genres": "Action, Adventure",
                "tags": "puzzle, co-op, funny, sci-fi",
                "description": "Solve physics puzzles with portals in a sharp and funny science fiction campaign.",
                "release_year": 2011,
                "positive_ratio": 98,
                "review_keywords": "funny, clever, memorable, polished",
                "sentiment_score": 0.96,
            },
            {
                "game_id": "mock_008",
                "title": "The Witcher 3: Wild Hunt",
                "genres": "RPG",
                "tags": "open world, fantasy, story rich, choices matter",
                "description": "Hunt monsters and make difficult choices across a large fantasy open world.",
                "release_year": 2015,
                "positive_ratio": 96,
                "review_keywords": "immersive, emotional, fantasy, story rich",
                "sentiment_score": 0.93,
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


def model_cache_mtime_ns() -> int:
    return MODEL_CACHE_PATH.stat().st_mtime_ns if MODEL_CACHE_PATH.exists() else 0


@st.cache_resource(show_spinner=False)
def get_ollama_explainer() -> OllamaExplainer:
    return OllamaExplainer()


@st.cache_resource(show_spinner=False)
def build_recommenders(
    games: pd.DataFrame,
    cache_version: int,
    persistent_cache_mtime_ns: int,
) -> tuple[ContentBasedRecommender, HybridRecommender, SemanticRecommender | None]:
    content_recommender = load_persistent_recommender(games, cache_version)
    if content_recommender is None:
        content_recommender = ContentBasedRecommender().fit(games)
        setattr(content_recommender, "cache_source", "sessao")
    else:
        setattr(content_recommender, "cache_source", "persistente")

    provider = create_embedding_provider()
    semantic_recommender = None
    if provider:
        semantic_recommender = SemanticRecommender(provider)
        try:
            semantic_recommender.fit(games)
        except Exception:
            semantic_recommender = None

    hybrid = HybridRecommender(
        content_recommender=content_recommender,
        semantic_recommender=semantic_recommender
    )
    return content_recommender, hybrid, semantic_recommender


def load_persistent_recommender(games: pd.DataFrame, cache_version: int) -> ContentBasedRecommender | None:
    if not MODEL_CACHE_PATH.exists():
        return None
    if GAMES_CSV_PATH.exists() and MODEL_CACHE_PATH.stat().st_mtime_ns < GAMES_CSV_PATH.stat().st_mtime_ns:
        return None

    try:
        with MODEL_CACHE_PATH.open("rb") as cache_file:
            candidate = pickle.load(cache_file)
    except Exception:
        return None

    if not isinstance(candidate, ContentBasedRecommender):
        return None
    if getattr(candidate, "model_cache_version", None) != cache_version:
        return None
    if not recommender_matches_games(candidate, games):
        return None
    return candidate


def recommender_matches_games(recommender: ContentBasedRecommender, games: pd.DataFrame) -> bool:
    cached_games = getattr(recommender, "games", None)
    if not isinstance(cached_games, pd.DataFrame) or len(cached_games) != len(games):
        return False
    if "game_id" not in cached_games.columns or "game_id" not in games.columns:
        return False

    cached_ids = cached_games["game_id"].astype(str).reset_index(drop=True)
    current_ids = games["game_id"].astype(str).reset_index(drop=True)
    return cached_ids.equals(current_ids)


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
    if "sentiment_score" in games.columns:
        games["sentiment_score"] = pd.to_numeric(games["sentiment_score"], errors="coerce")
    for column in ["url_store", "url_ref", "header_image", "steam_appid", "developer", "publisher", "review_keywords"]:
        if column in games.columns:
            games[column] = games[column].fillna("").astype(str)
    games["has_cover_image"] = games.apply(has_real_cover_image, axis=1)
    return games.reset_index(drop=True)


def split_terms(values: Iterable[str]) -> list[str]:
    terms: set[str] = set()
    for value in values:
        for term in re.split(r"[,|;/]", str(value)):
            normalized = term.strip()
            if normalized:
                terms.add(normalized)
    return sorted(terms, key=str.lower)


def top_terms(values: pd.Series, limit: int = 250) -> list[str]:
    counts: dict[str, int] = {}
    for value in values.dropna().astype(str):
        for term in split_terms([value]):
            counts[term] = counts.get(term, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))[:limit]]


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
    reference_game_ids: list[str],
    selected_genres: list[str],
    selected_tags: list[str],
    selected_developers: list[str],
    year_range: tuple[int, int],
    price_range: tuple[float, float] | None,
    min_positive_ratio: int,
) -> pd.DataFrame:
    filtered = games.copy()

    filtered = filtered[filtered["positive_ratio"] >= min_positive_ratio]
    filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]
    if price_range is not None and "price" in filtered.columns:
        filtered = filtered[filtered["price"].between(price_range[0], price_range[1])]

    if selected_genres:
        selected = {genre.lower() for genre in selected_genres}
        filtered = filtered[
            filtered["genres"].apply(
                lambda value: bool(selected.intersection({term.lower() for term in split_terms([value])}))
            )
        ]
    if selected_tags:
        selected = {tag.lower() for tag in selected_tags}
        filtered = filtered[
            filtered["tags"].apply(lambda value: bool(selected.intersection({term.lower() for term in split_terms([value])})))
        ]
    if selected_developers and "developer" in filtered.columns:
        selected = {developer.lower() for developer in selected_developers}
        filtered = filtered[
            filtered["developer"].apply(
                lambda value: bool(selected.intersection({term.lower() for term in split_terms([value])}))
            )
        ]

    scored = recommender.recommend(
        games=games,
        query=query,
        reference_game_ids=reference_game_ids or None,
        top_n=len(games),
    )
    filtered = scored.loc[filtered.index].copy()

    # Proxy de popularidade: jogos famosos tem keywords de reviews extraidas
    if "review_keywords" in filtered.columns:
        filtered["has_reviews"] = filtered["review_keywords"].astype(str).str.strip() != ""
    else:
        filtered["has_reviews"] = False

    has_active_intent = bool(normalize_text(query) or reference_game_ids)

    if not has_active_intent:
        # Evitar a anomalia do "100% de 1 review". Puxamos os 95~98% famosos pro topo.
        filtered["adjusted_ratio"] = filtered["positive_ratio"].apply(lambda x: x if x <= 97 else 97 - (x - 97))
        return filtered.sort_values(
            ["has_reviews", "adjusted_ratio", "release_year"],
            ascending=[False, False, False],
        )

    return filtered.sort_values(
        ["score", "has_reviews", "positive_ratio", "release_year"],
        ascending=[False, False, False, False],
    )


def term_set(value: str) -> set[str]:
    return {normalize_text(term) for term in split_terms([value]) if normalize_text(term)}


def query_term_matches(row: pd.Series, query: str) -> list[str]:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return []

    matches = []
    opinion_terms = row.get("review_keywords", "")
    for term in split_terms([f"{row['genres']}, {row['tags']}, {opinion_terms}"]):
        normalized_term = normalize_text(term)
        if normalized_term and normalized_term in normalized_query:
            matches.append(term)
    return matches[:6]


def reference_term_matches(row: pd.Series, reference_rows: list[pd.Series]) -> list[str]:
    if not reference_rows:
        return []

    row_terms = term_set(f"{row['genres']}, {row['tags']}")
    reference_terms: set[str] = set()
    for reference_row in reference_rows:
        reference_terms.update(term_set(f"{reference_row['genres']}, {reference_row['tags']}"))
    common = row_terms.intersection(reference_terms)
    labels = split_terms([f"{row['genres']}, {row['tags']}"])
    return [label for label in labels if normalize_text(label) in common][:6]


def build_explanation(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_rows: list[pd.Series],
) -> list[str]:
    reasons = []
    query_matches = query_term_matches(row, query)
    reference_matches = reference_term_matches(row, reference_rows)

    if query_matches:
        reasons.append(f"Termos da busca/opiniao encontrados: {', '.join(query_matches)}.")
    elif normalize_text(query):
        reasons.append("Descricoes, tags e termos de reviews ficaram proximos da sua busca pelo TF-IDF.")

    if reference_matches:
        reasons.append(f"Em comum com {reference_label}: {', '.join(reference_matches)}.")
    elif reference_rows:
        reasons.append(f"Similaridade textual com {reference_label} nas tags, generos e descricao.")

    if not reasons:
        reasons.append("Destaque no catálogo e aclamado pela comunidade.")

    if "sentiment_score_normalized" in row:
        reasons.append("Recomendação baseada no forte engajamento positivo e reviews recentes.")
    else:
        reasons.append(f"Selo de Aprovação: {row['positive_ratio']:.0f}% de avaliações positivas.")
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


def lucide_icon(name: str) -> str:
    return f'<span class="ludex-icon ludex-icon-{safe_html(name)}" aria-hidden="true"></span>'


def steam_appid_header_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return ""
    try:
        appid = str(int(float(raw)))
    except ValueError:
        return ""
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"


def resolve_header_image_url(row: pd.Series) -> str:
    raw_image_url = str(row.get("header_image", "") or "").strip()
    if raw_image_url and "placeholder.com" not in raw_image_url.lower():
        match = re.search(r"/steam/apps/(\d+)/", raw_image_url)
        if match:
            return steam_appid_header_url(match.group(1))
        image_url = safe_url(raw_image_url)
        if image_url:
            return image_url
    return safe_url(steam_appid_header_url(row.get("steam_appid", "")))


def has_real_cover_image(row: pd.Series) -> bool:
    return bool(resolve_header_image_url(row))


def header_image_html(row: pd.Series) -> str:
    image_url = resolve_header_image_url(row)
    title = safe_html(row["title"])
    if image_url:
        return (
            '<div class="ludex-card-media">'
            f'<img class="ludex-card-image" src="{image_url}" alt="" loading="lazy" referrerpolicy="no-referrer" '
            'onerror="this.style.display=\'none\';this.nextElementSibling.classList.add(\'visible\');">'
            f'<div class="ludex-card-image-fallback">{title}</div>'
            "</div>"
        )
    return (
        '<div class="ludex-card-media">'
        f'<div class="ludex-card-image-fallback visible">{title}</div>'
        "</div>"
    )


def action_links_html(row: pd.Series) -> str:
    store_url = safe_url(row.get("url_store", ""))
    ref_url = safe_url(row.get("url_ref", ""))
    links = []

    if store_url:
        links.append(
            f'<a class="ludex-action-link" href="{store_url}" target="_blank" rel="noopener noreferrer">'
            f"{lucide_icon('gamepad')} Steam</a>"
        )
    if ref_url:
        css_class = "ludex-action-link secondary" if store_url else "ludex-action-link"
        links.append(
            f'<a class="{css_class}" href="{ref_url}" target="_blank" rel="noopener noreferrer">'
            f"{lucide_icon('wrench')} PCGamingWiki</a>"
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


def sentiment_label(row: pd.Series) -> str:
    value = row.get("sentiment_score_normalized", None)
    if value is None or pd.isna(value):
        value = float(row["positive_ratio"]) / 100
    value = float(value)
    if value >= 0.82:
        return "muito positiva"
    if value >= 0.65:
        return "positiva"
    if value >= 0.45:
        return "dividida"
    return "critica"


def community_keywords(row: pd.Series, limit: int = 4) -> list[str]:
    source = row.get("review_keywords", "")
    if not str(source or "").strip():
        source = row.get("tags", "")
    return split_terms([source])[:limit]


def community_summary_html(row: pd.Series) -> str:
    keywords = community_keywords(row)
    if keywords:
        keyword_text = ", ".join(safe_html(keyword) for keyword in keywords)
        line = f"A comunidade descreve este jogo como {keyword_text}."
        badges = "".join(f'<span class="ludex-pill">{safe_html(keyword)}</span>' for keyword in keywords)
    else:
        line = "Ainda sem termos recorrentes de reviews; usando popularidade como sinal."
        badges = ""

    return (
        '<div class="ludex-community">'
        '<div class="ludex-community-title">Resumo da Comunidade</div>'
        f'<p class="ludex-community-line">Sentimento {safe_html(sentiment_label(row))}. {line}</p>'
        f'<div class="ludex-badges">{badges}</div>'
        "</div>"
    )


def score_breakdown_html(row: pd.Series) -> str:
    scores = [
        ("Final", float(row.get("score", 0.0))),
        ("Conteudo", float(row.get("content_score", 0.0))),
        ("Semantico", float(row.get("semantic_score", 0.0))),
        ("Busca", float(row.get("opinion_score", row.get("text_search_score", 0.0)))),
        ("Qualidade", float(row.get("quality_score", row.get("popularity_score", 0.0)))),
    ]
    boxes = []
    for label, value in scores:
        value = max(0.0, min(1.0, value))
        boxes.append(
            "\n".join(
                [
                    '<div class="ludex-score-box">',
                    f'<div class="ludex-score-label">{safe_html(label)}</div>',
                    f'<div class="ludex-score-value">{value:.2f}</div>',
                    f'<div class="ludex-score-track"><div class="ludex-score-fill" style="--score-width: {value * 100:.0f}%"></div></div>',
                    "</div>",
                ]
            )
        )
    return '<div class="ludex-score-row">' + "".join(boxes) + "</div>"


def game_card_html(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_rows: list[pd.Series],
) -> str:
    year = int(row["release_year"])
    year_label = str(year) if year > 0 else "N/D"
    description = safe_html(truncate_text(row["description"], max_chars=155))
    genre_badges = badges_html(row["genres"], limit=3, css_class="ludex-badge")
    tag_badges = badges_html(row["tags"], limit=3, css_class="ludex-pill")
    reasons = build_explanation(row, query, reference_label, reference_rows)

    return "\n".join(
        [
            '<article class="ludex-card">',
            header_image_html(row),
            '<div class="ludex-card-top">',
            f'<h3 class="ludex-game-title">{safe_html(row["title"])}</h3>',
            "</div>",
            '<div class="ludex-meta">',
            f'<span class="ludex-meta-chip">{lucide_icon("calendar")} {year_label}</span>',
            f'<span class="ludex-meta-chip">{lucide_icon("coins")} {safe_html(format_price(row))}</span>',
            f'<span class="ludex-meta-chip">{lucide_icon("thumbs-up")} {float(row["positive_ratio"]):.0f}%</span>',
            "</div>",
            f'<div class="ludex-badges">{genre_badges}</div>',
            f'<p class="ludex-description">{description}</p>',
            f'<div class="ludex-badges">{tag_badges}</div>',
            '<div class="ludex-why">',
            '<div class="ludex-why-title">Sinal principal</div>',
            reasons_html(reasons),
            "</div>",
            f'<div class="ludex-actions">{action_links_html(row)}</div>',
            "</article>",
        ]
    )


def render_game_card(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_rows: list[pd.Series],
) -> None:
    st.markdown(game_card_html(row, query, reference_label, reference_rows), unsafe_allow_html=True)

    game_id = str(row['game_id'])
    state_key = f"ia_explanation_{game_id}"

    # Renderiza botao fora da string HTML pois precisa de interatividade no backend Python
    if st.button("Por que este jogo?", key=f"btn_ia_{game_id}", use_container_width=True):
        with st.spinner("O agente está analisando..."):
            explainer = get_ollama_explainer()
            score = float(row["score"])
            resposta = explainer.generate_explanation(row, query, reference_label, score)
            st.session_state[state_key] = resposta

    if state_key in st.session_state:
        st.info(st.session_state[state_key])


def render_insights_panel(games: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    best_score = f"{recommendations['score'].max():.2f}" if not recommendations.empty else "0.00"
    required = "".join(f'<span class="ludex-pill">{safe_html(column)}</span>' for column in REQUIRED_COLUMNS)
    optional = "".join(
        f'<span class="ludex-pill">{safe_html(column)}</span>'
        for column in ["header_image", "review_keywords", "sentiment_score", "url_store", "url_ref"]
    )
    st.markdown(
        "\n".join(
            [
                '<aside class="ludex-side-panel">',
                '<h3 class="ludex-panel-title">Detalhes tecnicos</h3>',
                '<div class="ludex-panel-grid">',
                '<div class="ludex-panel-stat">',
                '<span class="ludex-panel-label">Catalogo</span>',
                f'<span class="ludex-panel-value">{len(games):,}</span>'.replace(",", "."),
                "</div>",
                '<div class="ludex-panel-stat">',
                '<span class="ludex-panel-label">Cards</span>',
                f'<span class="ludex-panel-value">{len(recommendations)}</span>',
                "</div>",
                '<div class="ludex-panel-stat">',
                '<span class="ludex-panel-label">Melhor score</span>',
                f'<span class="ludex-panel-value">{best_score}</span>',
                "</div>",
                "</div>",
                '<p class="ludex-panel-copy">Formula ativa: conteudo TF-IDF, busca opinativa e qualidade '
                "da comunidade com normalizacao dinamica dos sinais disponiveis.</p>",
                '<p class="ludex-panel-copy">Contrato obrigatorio</p>',
                f'<div class="ludex-contract">{required}</div>',
                '<p class="ludex-panel-copy" style="margin-top: 0.9rem;">Campos premium</p>',
                f'<div class="ludex-contract">{optional}</div>',
                "</aside>",
            ]
        ),
        unsafe_allow_html=True,
    )


def render_ai_status(content_recommender: ContentBasedRecommender, semantic_recommender: SemanticRecommender | None) -> None:
    max_features = int(getattr(content_recommender, "max_features", 30000))
    features_label = f"{max_features // 1000}k features" if max_features >= 1000 and max_features % 1000 == 0 else f"{max_features:,} features".replace(",", ".")
    ngram_range = getattr(content_recommender, "ngram_range", (1, 2))
    ngram_label = f"{int(ngram_range[0])}-{int(ngram_range[1])}"
    learned_vocabulary = len(getattr(getattr(content_recommender, "vectorizer", None), "vocabulary_", {}) or {})
    opinion_vocabulary = len(getattr(getattr(content_recommender, "opinion_vectorizer", None), "vocabulary_", {}) or {})
    cache_source = getattr(content_recommender, "cache_source", "sessao")
    semantic_label = semantic_recommender.provider.provider_id if semantic_recommender and semantic_recommender.provider else "Desligado"

    with st.expander("Detalhes tecnicos do ranking", expanded=False):
        st.markdown(
            "\n".join(
                [
                    '<div class="ludex-ai-grid">',
                    '<div class="ludex-ai-card">',
                    '<span class="ludex-ai-label">Vocabulario</span>',
                    f'<span class="ludex-ai-value">{safe_html(features_label)}</span>',
                    "</div>",
                    '<div class="ludex-ai-card">',
                    '<span class="ludex-ai-label">N-Grams</span>',
                    f'<span class="ludex-ai-value">{safe_html(ngram_label)}</span>',
                    "</div>",
                    '<div class="ludex-ai-card">',
                    '<span class="ludex-ai-label">Peso Tags</span>',
                    '<span class="ludex-ai-value">10x</span>',
                    "</div>",
                    '<div class="ludex-ai-card">',
                    '<span class="ludex-ai-label">Semantico</span>',
                    f'<span class="ludex-ai-value" title="{safe_html(semantic_label)}">{safe_html(semantic_label.split(":")[-1]) if semantic_recommender else "Off"}</span>',
                    "</div>",
                    "</div>",
                    '<p class="ludex-ai-note">',
                    f"TF-IDF treinado com {learned_vocabulary:,} termos de conteudo e ".replace(",", ".")
                    + f"{opinion_vocabulary:,} termos opinativos.".replace(",", "."),
                    "</p>",
                    '<p class="ludex-ai-note">',
                    f"Fonte do indice: cache {safe_html(cache_source)}.",
                    "</p>",
                ]
            ),
            unsafe_allow_html=True,
        )


def render_recommendation_grid(
    recommendations: pd.DataFrame,
    query: str,
    reference_label: str,
    reference_rows: list[pd.Series],
) -> None:
    for start in range(0, len(recommendations), 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, recommendations.iloc[start : start + 3].iterrows()):
            with column:
                render_game_card(row, query, reference_label, reference_rows)


def main() -> None:
    st.set_page_config(
        page_title="Ludex",
        page_icon="🕹️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    try:
        games = validate_contract(load_games(DATA_CACHE_VERSION, games_csv_mtime_ns()))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_hero(len(games))

    with st.spinner("Montando indice TF-IDF e Semantico (pode demorar na primeira vez)..."):
        content_recommender, hybrid_recommender, semantic_recommender = build_recommenders(
            games,
            RECOMMENDER_CACHE_VERSION,
            model_cache_mtime_ns(),
        )

    available_genres = split_terms(games["genres"])
    available_tags = top_terms(games["tags"], limit=160)
    available_developers = top_terms(games["developer"], limit=180) if "developer" in games.columns else []
    reference_options, reference_labels = resolve_reference_options(games)
    valid_years = games.loc[games["release_year"] > 0, "release_year"]
    min_year = int(valid_years.min()) if not valid_years.empty else 1980
    max_year = int(games["release_year"].max() or 2026)

    with st.container(key="ludex_search_panel"):
        st.markdown(
            f"""
            <div class="ludex-search-title">{lucide_icon("search")} Motor de recomendacao</div>
            <p class="ludex-search-copy">
                Escolha um ou mais jogos para montar seu perfil de gosto ou descreva a experiencia desejada.
            </p>
            """,
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "Intencao de busca",
            placeholder="Ex.: roguelike desafiador, fazenda relaxante, narrativa cyberpunk",
        )
        reference_game_ids = st.multiselect(
            "Perfil de gosto",
            options=[game_id for game_id in reference_options if game_id is not None],
            format_func=lambda game_id: reference_labels[game_id],
            placeholder="Selecione um ou mais jogos",
        )
        selected_reference_labels = [reference_labels[game_id] for game_id in reference_game_ids]
        if selected_reference_labels:
            reference_label = ", ".join(selected_reference_labels[:2])
            if len(selected_reference_labels) > 2:
                reference_label += f" +{len(selected_reference_labels) - 2}"
        else:
            reference_label = "Nenhum"

    with st.sidebar:
        render_ai_status(content_recommender, semantic_recommender)

        st.markdown(
            """
            <div class="ludex-sidebar-brand">
                <p class="ludex-sidebar-title">Filtros</p>
                <p class="ludex-sidebar-subtitle">Refine a vitrine por categoria, qualidade e estúdio.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("palette")} Estetica</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Generos e tags", expanded=True):
            selected_genres = st.multiselect("Generos", options=available_genres, placeholder="Escolha um ou mais...")
            selected_tags = st.multiselect("Tags reais do catalogo", options=available_tags, placeholder="Escolha um ou mais...")

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("joystick")} Gameplay</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Periodo e qualidade", expanded=True):
            if min_year < max_year:
                year_range = st.slider("Ano de lancamento", min_year, max_year, (min_year, max_year))
            else:
                year_range = (min_year, max_year)
                st.caption(f"Ano de lancamento: {min_year}")
            min_positive_ratio = st.slider("Reviews positivas minimas", 0, 100, 70)
            top_n = st.slider("Quantidade de cards", 3, 24, 9)

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("building-2")} Estudio</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Desenvolvedor", expanded=False):
            selected_developers = st.multiselect("Filtrar por desenvolvedor", options=available_developers, placeholder="Escolha um ou mais...")

        if games["game_id"].astype(str).str.startswith("mock_").all():
            source_label = "Dados mock ativos"
        else:
            source_label = "Dataset processado carregado"
        st.markdown(
            f'<div class="ludex-sidebar-status">{lucide_icon("database")} {safe_html(source_label)}</div>',
            unsafe_allow_html=True,
        )

    recommendations = filter_games(
        games=games,
        recommender=hybrid_recommender,
        query=query,
        reference_game_ids=reference_game_ids,
        selected_genres=selected_genres,
        selected_tags=selected_tags,
        selected_developers=selected_developers,
        year_range=year_range,
        price_range=None,
        min_positive_ratio=min_positive_ratio,
    ).head(top_n)

    reference_rows = []
    if reference_game_ids:
        reference_matches = games[games["game_id"].astype(str).isin({str(game_id) for game_id in reference_game_ids})]
        reference_rows = [row for _, row in reference_matches.iterrows()]

    # UI Principal dividida em Abas
    tab1, tab2 = st.tabs(["Vitrine", "Assistente Ludex"])

    with tab1:
        with st.expander("Detalhes tecnicos", expanded=False):
            render_insights_panel(games, recommendations)

        st.markdown(
            """
            <div class="ludex-grid-title">
                <h2>Top recomendacoes</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if reference_label != "Nenhum":
            st.markdown(
                f'<p class="ludex-active-query">Perfil ativo: {safe_html(reference_label)}</p>',
                unsafe_allow_html=True,
            )
        if normalize_text(query):
            st.markdown(
                f'<p class="ludex-active-query">Busca ativa: {safe_html(query)}</p>',
                unsafe_allow_html=True,
            )
        if recommendations.empty:
            st.markdown(
                """
                <div class="ludex-empty">
                    Nenhum jogo encontrado com os filtros atuais. Reduza ano, genero, tags ou reviews positivas.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_recommendation_grid(recommendations, query, reference_label, reference_rows)

    with tab2:
        st.markdown("### Assistente Ludex")
        explainer = get_ollama_explainer()
        if explainer.is_active:
            st.success(f"Ollama ativo com o modelo `{explainer.model_id}`. O assistente respondera usando o catalogo recuperado pelo Ludex.")
        else:
            st.info(
                "Ollama/modelo indisponivel. O assistente continua funcionando em modo local, "
                "com respostas deterministicas a partir do ranking TF-IDF/hibrido."
            )
            with st.expander("Como ativar o Ollama", expanded=False):
                st.code(
                    "ollama serve\n"
                    "ollama create ludex-assistant -f ludex-llama3-finetuned_gguf/Modelfile",
                    language="bash",
                )
                st.caption(explainer.status_message)
        st.markdown("Descreva o que quer jogar; o assistente busca candidatos no catalogo e responde em cima desses resultados.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Me recomende um jogo Cyberpunk"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("O agente está buscando no catálogo e pensando..."):
                    response = explainer.chat(prompt, st.session_state.messages[:-1], games, hybrid_recommender)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
