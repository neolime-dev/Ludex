from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import re

import numpy as np
import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender, normalize_ascii

if TYPE_CHECKING:
    from src.recommenders.semantic import SemanticRecommender


@dataclass(frozen=True)
class HybridWeights:
    content: float = 0.50
    semantic: float = 0.25
    opinion: float = 0.15
    quality: float = 0.10


QUERY_CONCEPTS = {
    "cyberpunk": ["cyberpunk", "dystopian"],
    "neon": ["neon"],
    "fazenda": ["farm", "farming", "agriculture", "crops", "farm sim", "farming sim"],
    "relaxante": ["relaxing", "cozy", "cosy", "laid back"],
    "roguelike": ["roguelike", "roguelite", "procedural", "randomized"],
    "desafiador": ["difficult", "challenging", "hard", "punishing"],
    "desafiadores": ["difficult", "challenging", "hard", "punishing"],
    "dificil": ["difficult", "challenging", "hard", "punishing"],
    "rpg": ["rpg", "role playing", "role-playing"],
    "narrativo": ["narrative", "story rich", "dialogue"],
    "narrativa": ["narrative", "story rich", "dialogue"],
    "escolhas": ["choices matter", "multiple endings", "choice", "choices"],
    "escolha": ["choices matter", "multiple endings", "choice", "choices"],
}

QUERY_STOPWORDS = {"a", "as", "com", "de", "do", "da", "e", "jogo", "o", "os", "para", "um", "uma"}


@dataclass
class HybridRecommender:
    content_recommender: ContentBasedRecommender
    semantic_recommender: "SemanticRecommender | None" = None
    weights: HybridWeights = HybridWeights()

    def score(
        self,
        games: pd.DataFrame,
        query: str = "",
        reference_game_ids: list[str] | None = None,
        reference_game_id: str | None = None,
    ) -> pd.DataFrame:
        reference_game_ids = self._normalize_reference_ids(reference_game_ids, reference_game_id)
        scored = games.reset_index(drop=True).copy()
        scored["content_score"] = self._align_scores_by_game_id(
            self._reference_scores(reference_game_ids),
            scored["game_id"],
        )
        scored["opinion_score"] = self._align_scores_by_game_id(
            self._opinion_scores(query),
            scored["game_id"],
        )
        scored["query_coverage_score"] = self._query_coverage_scores(scored, query)
        if str(query or "").strip():
            scored["opinion_score"] = scored["opinion_score"] * (0.35 + 0.65 * scored["query_coverage_score"])
        scored["text_search_score"] = scored["opinion_score"]
        scored["semantic_score"] = self._align_scores_by_game_id(
            self._semantic_scores(query=query, reference_game_ids=reference_game_ids),
            scored["game_id"],
        )
        scored["popularity_score"] = (scored["positive_ratio"] / 100).clip(0, 1)
        scored["sentiment_score_normalized"] = self._sentiment_scores(scored)
        scored["quality_score"] = self._quality_scores(scored)

        active_weights = self._active_weights(query=query, reference_game_id=reference_game_ids[0] if reference_game_ids else None)
        scored["score"] = (
            scored["content_score"] * active_weights["content"]
            + scored["semantic_score"] * active_weights["semantic"]
            + scored["opinion_score"] * active_weights["opinion"]
            + scored["quality_score"] * active_weights["quality"]
        )
        return scored

    def recommend(
        self,
        games: pd.DataFrame,
        query: str = "",
        reference_game_ids: list[str] | None = None,
        reference_game_id: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        scored = self.score(
            games=games,
            query=query,
            reference_game_ids=reference_game_ids,
            reference_game_id=reference_game_id,
        )
        return scored.sort_values(["score", "positive_ratio", "release_year"], ascending=False).head(top_n)

    def _reference_scores(self, reference_game_ids: list[str] | None) -> pd.Series:
        if not reference_game_ids:
            return self._zero_scores(name="content_score")
        return self.content_recommender.score_by_game_ids(reference_game_ids).rename("content_score")

    def _opinion_scores(self, query: str) -> pd.Series:
        if not str(query or "").strip():
            return self._zero_scores(name="opinion_score")
        return self.content_recommender.score_by_opinion_text(query).rename("opinion_score")

    def _semantic_scores(
        self,
        query: str = "",
        reference_game_ids: list[str] | None = None,
    ) -> pd.Series:
        if self.semantic_recommender is None or not getattr(self.semantic_recommender, "is_active", False):
            return self._zero_scores(name="semantic_score")

        if reference_game_ids:
            return self.semantic_recommender.score_by_game_ids(reference_game_ids).rename("semantic_score")
        if str(query or "").strip():
            return self.semantic_recommender.score_by_text(query).rename("semantic_score")
        return self._zero_scores(name="semantic_score")

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

    def _normalize_reference_ids(
        self,
        reference_game_ids: list[str] | None,
        reference_game_id: str | None,
    ) -> list[str] | None:
        if reference_game_ids:
            return [str(game_id) for game_id in reference_game_ids]
        if reference_game_id:
            return [str(reference_game_id)]
        return None

    def _active_weights(self, query: str, reference_game_id: str | None) -> dict[str, float]:
        has_semantic = (
            self.semantic_recommender is not None
            and getattr(self.semantic_recommender, "is_active", False)
        )
        has_intent = bool(reference_game_id or str(query or "").strip())

        raw = {
            "content": self.weights.content if reference_game_id else 0.0,
            "semantic": self.weights.semantic if (has_semantic and has_intent) else 0.0,
            "opinion": self.weights.opinion if str(query or "").strip() else 0.0,
            "quality": self.weights.quality,
        }

        if raw["content"] == 0.0 and raw["semantic"] == 0.0 and raw["opinion"] == 0.0:
            return {"content": 0.0, "semantic": 0.0, "opinion": 0.0, "quality": 1.0}

        total = sum(raw.values())
        return {name: weight / total for name, weight in raw.items()}

    def _query_coverage_scores(self, games: pd.DataFrame, query: str) -> pd.Series:
        concepts = _query_concepts(query)
        if not concepts:
            return pd.Series(np.zeros(len(games)), index=games.index, name="query_coverage_score")

        searchable = (
            _safe_text_column(games, "title")
            + " "
            + _safe_text_column(games, "genres")
            + " "
            + _safe_text_column(games, "tags")
            + " "
            + _safe_text_column(games, "description")
            + " "
            + _safe_text_column(games, "review_keywords")
        ).map(lambda value: normalize_ascii(value).lower())

        scores = []
        for text in searchable:
            matched = 0
            for variants in concepts:
                if any(variant in text for variant in variants):
                    matched += 1
            scores.append(matched / len(concepts))
        return pd.Series(scores, index=games.index, name="query_coverage_score")


def _query_concepts(query: str) -> list[list[str]]:
    normalized = normalize_ascii(query).lower()
    tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token not in QUERY_STOPWORDS]
    concepts: list[list[str]] = []
    for token in tokens:
        variants = QUERY_CONCEPTS.get(token)
        if variants is None:
            variants = [token[:-1] if token.endswith("s") and len(token) > 3 else token, token]
        normalized_variants = sorted({normalize_ascii(variant).lower() for variant in variants if variant})
        if normalized_variants:
            concepts.append(normalized_variants)
    return concepts


def _safe_text_column(games: pd.DataFrame, column: str) -> pd.Series:
    if column not in games.columns:
        return pd.Series([""] * len(games), index=games.index)
    return games[column].fillna("").astype(str)
