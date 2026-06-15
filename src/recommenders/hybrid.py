from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender


@dataclass(frozen=True)
class HybridWeights:
    content: float = 0.5
    text_search: float = 0.3
    popularity: float = 0.2


@dataclass
class HybridRecommender:
    content_recommender: ContentBasedRecommender
    weights: HybridWeights = HybridWeights()

    def score(
        self,
        games: pd.DataFrame,
        query: str = "",
        reference_game_id: str | None = None,
    ) -> pd.DataFrame:
        scored = games.reset_index(drop=True).copy()
        scored["content_score"] = self._reference_scores(reference_game_id)
        scored["text_search_score"] = self._text_scores(query)
        scored["popularity_score"] = (scored["positive_ratio"] / 100).clip(0, 1)

        active_weights = self._active_weights(query=query, reference_game_id=reference_game_id)
        scored["score"] = (
            scored["content_score"] * active_weights["content"]
            + scored["text_search_score"] * active_weights["text_search"]
            + scored["popularity_score"] * active_weights["popularity"]
        )
        return scored

    def recommend(
        self,
        games: pd.DataFrame,
        query: str = "",
        reference_game_id: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        scored = self.score(games=games, query=query, reference_game_id=reference_game_id)
        return scored.sort_values(["score", "positive_ratio", "release_year"], ascending=False).head(top_n)

    def _reference_scores(self, reference_game_id: str | None) -> pd.Series:
        if not reference_game_id:
            return self._zero_scores(name="content_score")
        return self.content_recommender.score_by_game_id(reference_game_id).rename("content_score")

    def _text_scores(self, query: str) -> pd.Series:
        if not str(query or "").strip():
            return self._zero_scores(name="text_search_score")
        return self.content_recommender.score_by_text(query).rename("text_search_score")

    def _zero_scores(self, name: str) -> pd.Series:
        games = self.content_recommender.games
        return pd.Series(np.zeros(len(games)), index=games.index, name=name)

    def _active_weights(self, query: str, reference_game_id: str | None) -> dict[str, float]:
        raw = {
            "content": self.weights.content if reference_game_id else 0.0,
            "text_search": self.weights.text_search if str(query or "").strip() else 0.0,
            "popularity": self.weights.popularity,
        }

        if raw["content"] == 0.0 and raw["text_search"] == 0.0:
            return {"content": 0.0, "text_search": 0.0, "popularity": 1.0}

        total = sum(raw.values())
        return {name: weight / total for name, weight in raw.items()}
