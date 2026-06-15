from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender


@dataclass(frozen=True)
class HybridWeights:
    content: float = 0.4
    opinion: float = 0.3
    quality: float = 0.3


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
        scored["content_score"] = self._align_scores_by_game_id(
            self._reference_scores(reference_game_id),
            scored["game_id"],
        )
        scored["opinion_score"] = self._align_scores_by_game_id(
            self._opinion_scores(query),
            scored["game_id"],
        )
        scored["text_search_score"] = scored["opinion_score"]
        scored["popularity_score"] = (scored["positive_ratio"] / 100).clip(0, 1)
        scored["sentiment_score_normalized"] = self._sentiment_scores(scored)
        scored["quality_score"] = self._quality_scores(scored)

        active_weights = self._active_weights(query=query, reference_game_id=reference_game_id)
        scored["score"] = (
            scored["content_score"] * active_weights["content"]
            + scored["opinion_score"] * active_weights["opinion"]
            + scored["quality_score"] * active_weights["quality"]
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

    def _opinion_scores(self, query: str) -> pd.Series:
        if not str(query or "").strip():
            return self._zero_scores(name="opinion_score")
        return self.content_recommender.score_by_opinion_text(query).rename("opinion_score")

    def _sentiment_scores(self, games: pd.DataFrame) -> pd.Series:
        if "sentiment_score" not in games.columns:
            return games["popularity_score"].rename("sentiment_score_normalized")

        raw = pd.to_numeric(games["sentiment_score"], errors="coerce")
        if raw.notna().sum() == 0:
            return games["popularity_score"].rename("sentiment_score_normalized")

        filled = raw.fillna(raw.median())
        min_value = float(filled.min())
        max_value = float(filled.max())

        if min_value < 0 and max_value <= 1:
            normalized = (filled + 1) / 2
        elif max_value > 1:
            divisor = 5 if max_value <= 5 else 100
            normalized = filled / divisor
        else:
            normalized = filled

        return normalized.clip(0, 1).rename("sentiment_score_normalized")

    def _quality_scores(self, games: pd.DataFrame) -> pd.Series:
        if "sentiment_score" not in games.columns:
            return games["popularity_score"].rename("quality_score")
        quality = (0.55 * games["popularity_score"]) + (0.45 * games["sentiment_score_normalized"])
        return quality.clip(0, 1).rename("quality_score")

    def _zero_scores(self, name: str) -> pd.Series:
        games = self.content_recommender.games
        return pd.Series(np.zeros(len(games)), index=games.index, name=name)

    def _align_scores_by_game_id(self, scores: pd.Series, game_ids: pd.Series) -> pd.Series:
        scored = pd.Series(
            scores.to_numpy(),
            index=self.content_recommender.games["game_id"].astype(str),
            name=scores.name,
        )
        return game_ids.astype(str).map(scored).fillna(0.0)

    def _active_weights(self, query: str, reference_game_id: str | None) -> dict[str, float]:
        raw = {
            "content": self.weights.content if reference_game_id else 0.0,
            "opinion": self.weights.opinion if str(query or "").strip() else 0.0,
            "quality": self.weights.quality,
        }

        if raw["content"] == 0.0 and raw["opinion"] == 0.0:
            return {"content": 0.0, "opinion": 0.0, "quality": 1.0}

        total = sum(raw.values())
        return {name: weight / total for name, weight in raw.items()}
