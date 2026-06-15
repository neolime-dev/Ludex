from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAMES_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"

REQUIRED_COLUMNS = [
    "game_id",
    "title",
    "genres",
    "tags",
    "description",
    "release_year",
    "positive_ratio",
]


def load_games(path: str | Path = DEFAULT_GAMES_PATH) -> pd.DataFrame:
    games_path = Path(path)
    if not games_path.exists():
        raise FileNotFoundError(f"Arquivo de jogos nao encontrado: {games_path}")
    if games_path.stat().st_size <= 1:
        raise ValueError(f"Arquivo de jogos vazio: {games_path}")

    games = pd.read_csv(games_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in games.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Contrato de dados incompleto em {games_path}. Colunas ausentes: {missing}")

    return games
