from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEXT_COLUMNS = ["title", "genres", "tags", "description"]
OPINION_COLUMNS = ["review_keywords"]

QUERY_EXPANSIONS = {
    "acao": "action fast-paced combat",
    "aventura": "adventure exploration",
    "cartas": "card deckbuilding",
    "cooperativo": "co-op cooperative multiplayer",
    "criacao": "crafting building",
    "desafiador": "difficult challenging hard",
    "desafiadores": "difficult challenging hard",
    "dificil": "difficult challenging hard",
    "divertido": "fun funny enjoyable comedy",
    "emocionante": "emotional moving touching story rich",
    "emocional": "emotional moving touching story rich",
    "enjoativo": "boring repetitive tedious",
    "envolvente": "immersive engaging atmospheric",
    "fantasia": "fantasy",
    "fazenda": "farming life sim relaxing",
    "frustrante": "frustrating difficult unfair rage",
    "historia": "story rich narrative choices matter",
    "humor": "funny comedy",
    "imersiva": "immersive atmospheric deep",
    "imersivo": "immersive atmospheric deep",
    "mitologia": "mythology fantasy",
    "mundo": "world open world",
    "narrativa": "story rich narrative choices matter",
    "nostalgico": "nostalgic retro classic old school",
    "obra": "masterpiece great excellent",
    "quebra": "puzzle",
    "rapido": "fast-paced action",
    "recompensador": "rewarding satisfying progression",
    "relaxante": "relaxing casual cozy",
    "repetitivo": "repetitive grind tedious",
    "roguelike": "roguelike roguelite",
    "rpg": "rpg role-playing",
    "rpgs": "rpg role-playing",
    "satisfatorio": "satisfying rewarding",
    "simulacao": "simulation sim",
    "sombria": "dark dark fantasy",
    "sombrio": "dark dark fantasy",
    "tatica": "tactical strategy turn-based",
    "taticas": "tactical strategy turn-based",
    "tatico": "tactical strategy turn-based",
    "taticos": "tactical strategy turn-based",
    "viciante": "addictive replayable replay value progression",
}


@dataclass
class ContentBasedRecommender:
    max_features: int = 30000
    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)

    def fit(self, games: pd.DataFrame) -> "ContentBasedRecommender":
        self.games = games.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        self.matrix = self.vectorizer.fit_transform(self._build_corpus(self.games))
        self.opinion_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        self.opinion_matrix = self.opinion_vectorizer.fit_transform(self._build_opinion_corpus(self.games))
        self.game_id_to_index = {
            str(game_id): index for index, game_id in enumerate(self.games["game_id"].astype(str))
        }
        return self

    def score_by_text(self, query: str) -> pd.Series:
        self._ensure_fitted()
        if not str(query or "").strip():
            return self._empty_scores()

        query_vector = self.vectorizer.transform([expand_query(query)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        return pd.Series(scores, index=self.games.index, name="content_score")

    def score_by_opinion_text(self, query: str) -> pd.Series:
        self._ensure_fitted()
        if not str(query or "").strip():
            return self._empty_scores().rename("opinion_score")

        query_vector = self.opinion_vectorizer.transform([expand_query(query)])
        scores = cosine_similarity(query_vector, self.opinion_matrix).ravel()
        return pd.Series(scores, index=self.games.index, name="opinion_score")

    def score_by_game_id(self, game_id: str) -> pd.Series:
        self._ensure_fitted()
        index = self.game_id_to_index.get(str(game_id))
        if index is None:
            return self._empty_scores()

        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        scores[index] = 0.0
        return pd.Series(scores, index=self.games.index, name="content_score")

    def recommend_by_text(self, query: str, top_n: int = 10) -> pd.DataFrame:
        return self._rank(self.score_by_text(query), top_n=top_n)

    def recommend_similar_games(self, game_id: str, top_n: int = 10) -> pd.DataFrame:
        return self._rank(self.score_by_game_id(game_id), top_n=top_n)

    def _rank(self, scores: pd.Series, top_n: int) -> pd.DataFrame:
        results = self.games.copy()
        results["content_score"] = scores.to_numpy()
        return results.sort_values("content_score", ascending=False).head(top_n)

    def empty_scores(self, name: str = "content_score") -> pd.Series:
        return self._empty_scores().rename(name)

    def _empty_scores(self) -> pd.Series:
        return pd.Series(np.zeros(len(self.games)), index=self.games.index, name="content_score")

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "matrix") or not hasattr(self, "opinion_matrix"):
            raise RuntimeError("ContentBasedRecommender precisa ser treinado com .fit(games).")

    @staticmethod
    def _build_corpus(games: pd.DataFrame) -> pd.Series:
        text = _safe_text_frame(games, TEXT_COLUMNS)
        return (
            text["title"]
            + " "
            + (text["genres"] + " ") * 3
            + (text["tags"] + " ") * 3
            + text["description"]
        )

    @staticmethod
    def _build_opinion_corpus(games: pd.DataFrame) -> pd.Series:
        text = _safe_text_frame(games, TEXT_COLUMNS + OPINION_COLUMNS)
        return (
            text["title"]
            + " "
            + (text["genres"] + " ") * 2
            + (text["tags"] + " ") * 2
            + text["description"]
            + " "
            + (text["review_keywords"] + " ") * 5
        )


def _safe_text_frame(games: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        if column in games.columns:
            data[column] = games[column].fillna("").astype(str)
        else:
            data[column] = pd.Series([""] * len(games), index=games.index)
    return pd.DataFrame(data, index=games.index)


def expand_query(query: str) -> str:
    normalized = normalize_ascii(query)
    tokens = re.findall(r"[a-zA-Z0-9]+", normalized.lower())
    expansions = []
    for token in tokens:
        expansions.append(token)
        if token.endswith("s") and len(token) > 3:
            expansions.append(token[:-1])
        mapped = QUERY_EXPANSIONS.get(token)
        if mapped:
            expansions.append(mapped)
    return " ".join(expansions)


def normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in normalized if not unicodedata.combining(character))
