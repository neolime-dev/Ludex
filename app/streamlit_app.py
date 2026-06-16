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

SEARCH_EXAMPLES = [
    "Quero um jogo emocionante e imersivo",
    "Procuro algo viciante com muita progressao",
    "Evite jogos frustrantes, quero algo recompensador",
    "Sou fa de RPGs taticos e desafiadores",
    "Quero um roguelike rapido com progressao constante",
    "Procuro jogo relaxante de simulacao e crafting",
    "Gosto de fantasia sombria com historia forte",
    "Quero puzzle cooperativo com humor",
]
RECOMMENDER_CACHE_VERSION = 4
DATA_CACHE_VERSION = 3
GAMES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

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

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(56, 189, 248, 0.08), transparent 28rem),
                linear-gradient(210deg, rgba(34, 197, 94, 0.08), transparent 24rem),
                radial-gradient(circle at 50% -10%, rgba(39, 39, 42, 0.95), transparent 34rem),
                linear-gradient(180deg, #09090b 0%, #0b1120 46%, #09090b 100%);
            color: var(--ludex-text);
        }

        header[data-testid="stHeader"] {
            background: rgba(9, 9, 11, 0.76);
            border-bottom: 1px solid var(--ludex-line);
            backdrop-filter: blur(18px);
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(24, 24, 27, 0.98), rgba(9, 9, 11, 0.98));
            border-right: 1px solid var(--ludex-line);
        }

        section[data-testid="stSidebar"] * {
            color: var(--ludex-text);
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: rgba(9, 9, 11, 0.92);
            border: 1px solid var(--ludex-line-strong);
            color: var(--ludex-text);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }

        h1, h2, h3 {
            color: var(--ludex-text);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(39, 39, 42, 0.88), rgba(24, 24, 27, 0.94));
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.82rem 0.9rem;
            box-shadow: var(--ludex-shadow);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--ludex-muted);
            font-size: 0.78rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ludex-text);
            font-size: 1.28rem;
            font-weight: 800;
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

        .ludex-hero::after {
            content: "";
            position: absolute;
            inset: auto 1.2rem 1.2rem auto;
            width: 11rem;
            height: 11rem;
            border: 1px solid rgba(250, 250, 250, 0.08);
            border-radius: 999px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 68%);
            pointer-events: none;
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

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: var(--ludex-soft) !important;
            font-size: 0.78rem;
            font-weight: 760;
        }

        section[data-testid="stSidebar"] details {
            border: 1px solid var(--ludex-line);
            border-radius: 8px;
            padding: 0.2rem 0.72rem 0.72rem;
            background: rgba(9, 9, 11, 0.38);
        }

        section[data-testid="stSidebar"] details summary {
            color: var(--ludex-text);
            font-weight: 850;
        }

        section[data-testid="stSidebar"] [data-baseweb="slider"] > div {
            color: var(--ludex-accent);
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
            justify-content: space-between;
            gap: 0.9rem;
        }

        .ludex-game-title {
            color: var(--ludex-text);
            font-size: 1.05rem;
            line-height: 1.22;
            font-weight: 850;
            margin: 0;
        }

        .ludex-relevance {
            flex: 0 0 auto;
            min-width: 4rem;
            color: #09090b;
            background: linear-gradient(135deg, var(--ludex-warn), #f97316);
            border-radius: 6px;
            padding: 0.42rem 0.52rem;
            text-align: center;
            box-shadow: 0 12px 28px rgba(250, 204, 21, 0.18);
        }

        .ludex-relevance-value {
            display: block;
            font-size: 0.98rem;
            font-weight: 900;
            line-height: 1;
        }

        .ludex-relevance-label {
            display: block;
            margin-top: 0.14rem;
            font-size: 0.52rem;
            font-weight: 800;
            line-height: 1;
            text-transform: uppercase;
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
            grid-template-columns: repeat(3, minmax(0, 1fr));
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

        .stButton > button,
        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #38bdf8, #22c55e);
            color: #ffffff;
            border: 0;
            border-radius: 6px;
            font-weight: 800;
        }

        .stAlert {
            background: rgba(39, 39, 42, 0.82);
            border: 1px solid var(--ludex-line);
            color: var(--ludex-soft);
        }

        @media (max-width: 900px) {
            .ludex-card {
                min-height: auto;
            }
            .ludex-title {
                font-size: 2.1rem;
            }
            .ludex-score-row {
                grid-template-columns: 1fr;
            }
            .ludex-panel-grid {
                grid-template-columns: 1fr;
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
            <div class="ludex-kicker">Ludex Recommender | Premium NLP Build</div>
            <h1 class="ludex-title">Ludex</h1>
            <p class="ludex-subtitle">
                Uma vitrine curada com {total_label} titulos, ranking hibrido por similaridade,
                intencao de busca, sinais opinativos e qualidade da comunidade.
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
    reference_game_id: str | None,
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

    scored = recommender.score(games=games, query=query, reference_game_id=reference_game_id)
    filtered = scored.loc[filtered.index].copy()
    has_active_intent = bool(normalize_text(query) or reference_game_id)
    if not has_active_intent and "has_cover_image" in filtered.columns and filtered["has_cover_image"].sum() >= 12:
        filtered = filtered[filtered["has_cover_image"]].copy()
    return filtered.sort_values(
        ["score", "has_cover_image", "positive_ratio", "release_year"],
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
        reasons.append(f"Termos da busca/opiniao encontrados: {', '.join(query_matches)}.")
    elif normalize_text(query):
        reasons.append("Descricoes, tags e termos de reviews ficaram proximos da sua busca pelo TF-IDF.")

    if reference_matches:
        reasons.append(f"Em comum com {reference_label}: {', '.join(reference_matches)}.")
    elif reference_row is not None:
        reasons.append(f"Similaridade textual com {reference_label} nas tags, generos e descricao.")

    if not reasons:
        reasons.append("Sem busca ou jogo de referencia, usamos popularidade como fallback.")

    if "sentiment_score_normalized" in row:
        reasons.append(f"Sentimento da comunidade ponderado: {float(row['sentiment_score_normalized']):.2f}.")
    else:
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


def game_card_html(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_row: pd.Series | None,
) -> str:
    year = int(row["release_year"])
    year_label = str(year) if year > 0 else "N/D"
    score = float(row["score"])
    score_pct = max(0, min(100, int(round(score * 100))))
    opinion_score = float(row.get("opinion_score", row.get("text_search_score", 0.0)))
    community_score = float(row.get("quality_score", row.get("popularity_score", 0.0)))
    description = safe_html(truncate_text(row["description"], max_chars=155))
    genre_badges = badges_html(row["genres"], limit=3, css_class="ludex-badge")
    tag_badges = badges_html(row["tags"], limit=3, css_class="ludex-pill")
    reasons = build_explanation(row, query, reference_label, reference_row)

    return "\n".join(
        [
            '<article class="ludex-card">',
            header_image_html(row),
            '<div class="ludex-card-top">',
            f'<h3 class="ludex-game-title">{safe_html(row["title"])}</h3>',
            '<div class="ludex-relevance">',
            f'<span class="ludex-relevance-value">{score_pct}</span>',
            '<span class="ludex-relevance-label">match</span>',
            "</div>",
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
            '<div class="ludex-score-track">',
            f'<div class="ludex-score-fill" style="--score-width: {score_pct}%;"></div>',
            "</div>",
            '<details class="ludex-details">',
            '<summary>Analise completa</summary>',
            community_summary_html(row),
            '<div class="ludex-score-row">',
            '<div class="ludex-score-box">',
            '<div class="ludex-score-label">Final</div>',
            f'<div class="ludex-score-value">{score:.2f}</div>',
            "</div>",
            '<div class="ludex-score-box">',
            '<div class="ludex-score-label">Opiniao</div>',
            f'<div class="ludex-score-value">{opinion_score:.2f}</div>',
            "</div>",
            '<div class="ludex-score-box">',
            '<div class="ludex-score-label">Comunidade</div>',
            f'<div class="ludex-score-value">{community_score:.2f}</div>',
            "</div>",
            "</div>",
            "</details>",
            f'<div class="ludex-actions">{action_links_html(row)}</div>',
            "</article>",
        ]
    )


def render_game_card(
    row: pd.Series,
    query: str,
    reference_label: str,
    reference_row: pd.Series | None,
) -> None:
    st.markdown(game_card_html(row, query, reference_label, reference_row), unsafe_allow_html=True)


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
                '<h3 class="ludex-panel-title">Painel de Controle</h3>',
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
                '<span class="ludex-panel-label">Score</span>',
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
    available_tags = top_terms(games["tags"], limit=160)
    available_developers = top_terms(games["developer"], limit=180) if "developer" in games.columns else []
    reference_options, reference_labels = resolve_reference_options(games)
    valid_years = games.loc[games["release_year"] > 0, "release_year"]
    min_year = int(valid_years.min()) if not valid_years.empty else 1980
    max_year = int(games["release_year"].max() or 2026)
    price_range = None
    price_values = games["price"].dropna() if "price" in games.columns else pd.Series(dtype=float)
    max_price = float(price_values.max()) if not price_values.empty else 0.0

    with st.sidebar:
        st.markdown(
            """
            <div class="ludex-sidebar-brand">
                <p class="ludex-sidebar-title">Ludex</p>
                <p class="ludex-sidebar-subtitle">Controle fino para explorar o catalogo por gosto, preco e estudio.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("search")} Perfil</div>',
            unsafe_allow_html=True,
        )
        reference_game_id = st.selectbox(
            "Jogo de referencia",
            options=reference_options,
            format_func=lambda game_id: reference_labels[game_id],
        )
        reference_label = reference_labels[reference_game_id]
        selected_example = st.selectbox("Busca guiada", options=[""] + SEARCH_EXAMPLES)
        custom_query = st.text_input("Preferencia livre", placeholder="Ex.: roguelike dificil com boa historia")
        query = custom_query.strip() or selected_example

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("palette")} Estetica</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Generos e tags", expanded=True):
            selected_genres = st.multiselect("Generos", options=available_genres)
            selected_tags = st.multiselect("Tags de estilo", options=available_tags)

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
            f'<div class="ludex-filter-heading">{lucide_icon("badge-dollar-sign")} Preco</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Faixa real", expanded=True):
            if max_price > 0:
                price_ceiling = float(round(max_price))
                price_range = st.slider(
                    "Preco em USD",
                    min_value=0.0,
                    max_value=price_ceiling,
                    value=(0.0, price_ceiling),
                    step=1.0,
                    format="$%.0f",
                )
            else:
                st.caption("Sem precos numericos no dataset atual.")

        st.markdown(
            f'<div class="ludex-filter-heading">{lucide_icon("building-2")} Estudio</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Desenvolvedor", expanded=False):
            selected_developers = st.multiselect("Filtrar por desenvolvedor", options=available_developers)

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
        reference_game_id=reference_game_id,
        selected_genres=selected_genres,
        selected_tags=selected_tags,
        selected_developers=selected_developers,
        year_range=year_range,
        price_range=price_range,
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
        render_insights_panel(games, recommendations)


if __name__ == "__main__":
    main()
