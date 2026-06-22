"""Recomendador semantico baseado em embeddings densos.

Este modulo funciona em paralelo ao TF-IDF existente. Quando um
``EmbeddingProvider`` esta disponivel, gera/carrega embeddings do
catalogo e calcula similaridade por cosseno no espaco vetorial denso.

Quando nenhum provider esta disponivel, todos os metodos de score
retornam zeros — comportamento neutro que nao afeta o ranking hibrido.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommenders.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "cache"
EMBEDDINGS_FILE = "embeddings_catalog.npz"
EMBEDDINGS_META_FILE = "embeddings_meta.json"
SEMANTIC_CORPUS_VERSION = 1


@dataclass
class SemanticRecommender:
    """Recomendador baseado em embeddings semanticos.

    Se ``provider`` for ``None``, o recomendador opera em modo stub:
    ``fit()`` nao faz nada e todos os metodos de score retornam zeros.
    Isso permite que o ``HybridRecommender`` o aceite opcionalmente
    sem alterar o comportamento do ranking quando nao ha embeddings.
    """

    provider: EmbeddingProvider | None = None
    cache_dir: Path = field(default_factory=lambda: SEMANTIC_CACHE_DIR)

    def fit(self, games: pd.DataFrame) -> "SemanticRecommender":
        """Gera ou carrega embeddings do catalogo.

        Se o provider for None, apenas armazena o DataFrame para
        que os metodos de score possam retornar Series do tamanho certo.
        """
        self.games = games.reset_index(drop=True).copy()
        self.game_id_to_index: dict[str, int] = {
            str(game_id): idx
            for idx, game_id in enumerate(self.games["game_id"].astype(str))
        }

        if self.provider is None:
            self.embeddings = None
            logger.info("SemanticRecommender: sem provider, modo stub ativo.")
            return self

        cached = self._load_cache()
        if cached is not None:
            self.embeddings = cached
            logger.info(
                "SemanticRecommender: cache carregado (%d jogos, %d dims).",
                cached.shape[0],
                cached.shape[1],
            )
            return self

        corpus = self._build_semantic_corpus(self.games)
        logger.info("SemanticRecommender: gerando embeddings para %d jogos...", len(corpus))
        start = time.perf_counter()
        self.embeddings = self.provider.embed_texts(corpus)
        duration = time.perf_counter() - start
        logger.info("SemanticRecommender: embeddings gerados em %.2fs.", duration)

        self._save_cache(self.embeddings)
        return self

    def score_by_text(self, query: str) -> pd.Series:
        """Calcula similaridade semantica entre a query e o catalogo.

        Retorna zeros se o provider nao estiver ativo ou se a query
        estiver vazia.
        """
        self._ensure_fitted()
        if self.embeddings is None or not str(query or "").strip():
            return self._zero_scores()

        query_embedding = self.provider.embed_query(query)
        scores = (self.embeddings @ query_embedding.T).ravel()
        return pd.Series(scores, index=self.games.index, name="semantic_score")

    def score_by_game_ids(self, game_ids: list[str]) -> pd.Series:
        """Calcula similaridade semantica usando centroide dos jogos de referencia.

        Retorna zeros se o provider nao estiver ativo ou se nenhum
        game_id for valido.
        """
        self._ensure_fitted()
        if self.embeddings is None:
            return self._zero_scores()

        valid_indices = [
            self.game_id_to_index[str(gid)]
            for gid in game_ids
            if str(gid) in self.game_id_to_index
        ]

        if not valid_indices:
            return self._zero_scores()

        centroid = self.embeddings[valid_indices].mean(axis=0, keepdims=True)
        # Re-normalizar o centroide para manter produto escalar = cosseno
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        scores = (self.embeddings @ centroid.T).ravel()

        # Zerar scores dos proprios jogos de referencia
        for idx in valid_indices:
            scores[idx] = 0.0

        return pd.Series(scores, index=self.games.index, name="semantic_score")

    @property
    def is_active(self) -> bool:
        """True se o recomendador tem embeddings carregados."""
        return self.embeddings is not None

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    @staticmethod
    def _build_semantic_corpus(games: pd.DataFrame) -> list[str]:
        """Constroi um texto por jogo otimizado para embedding.

        Diferente do TF-IDF, embeddings nao precisam de repeticao de
        pesos. O formato e simples e descritivo para maximizar a
        qualidade da representacao vetorial densa.
        """
        corpus: list[str] = []
        for _, row in games.iterrows():
            title = str(row.get("title", "")).strip()
            genres = str(row.get("genres", "")).strip()
            tags = str(row.get("tags", "")).strip()
            description = str(row.get("description", "")).strip()

            parts = [title]
            if genres:
                parts.append(f"Genres: {genres}.")
            if tags:
                # Limpar separadores para texto natural
                clean_tags = re.sub(r"[|;/]", ",", tags)
                parts.append(f"Tags: {clean_tags}.")
            if description:
                parts.append(description)

            corpus.append(" ".join(parts))
        return corpus

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _game_ids_hash(self) -> str:
        """Hash SHA-256 dos game_ids para validacao de cache."""
        ids_str = ",".join(self.games["game_id"].astype(str))
        return hashlib.sha256(ids_str.encode("utf-8")).hexdigest()

    def _load_cache(self) -> np.ndarray | None:
        """Carrega embeddings do cache se valido."""
        if self.provider is None:
            return None

        meta_path = self.cache_dir / EMBEDDINGS_META_FILE
        data_path = self.cache_dir / EMBEDDINGS_FILE

        if not meta_path.exists() or not data_path.exists():
            return None

        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
        except Exception:
            return None

        # Validar provider
        if meta.get("provider_id") != self.provider.provider_id:
            logger.info("Cache invalidado: provider mudou (%s -> %s).", meta.get("provider_id"), self.provider.provider_id)
            return None

        # Validar corpus version
        if meta.get("corpus_version") != SEMANTIC_CORPUS_VERSION:
            logger.info("Cache invalidado: corpus_version mudou.")
            return None

        # Validar game_ids
        if meta.get("game_ids_hash") != self._game_ids_hash():
            logger.info("Cache invalidado: game_ids mudaram.")
            return None

        # Validar dimensao
        if meta.get("dimension") != self.provider.dimension:
            logger.info("Cache invalidado: dimensao mudou.")
            return None

        # Validar games.csv mtime
        games_csv = Path(__file__).resolve().parents[2] / "data" / "processed" / "games.csv"
        if games_csv.exists():
            csv_mtime = games_csv.stat().st_mtime_ns
            if meta.get("games_csv_mtime_ns", 0) < csv_mtime:
                logger.info("Cache invalidado: games.csv mais novo que cache.")
                return None

        try:
            data = np.load(data_path)
            embeddings = data["embeddings"]
        except Exception:
            return None

        if embeddings.shape[0] != len(self.games):
            logger.info("Cache invalidado: tamanho nao bate (%d vs %d).", embeddings.shape[0], len(self.games))
            return None

        return embeddings

    def _save_cache(self, embeddings: np.ndarray) -> None:
        """Persiste embeddings e metadados em disco."""
        if self.provider is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data_path = self.cache_dir / EMBEDDINGS_FILE
        meta_path = self.cache_dir / EMBEDDINGS_META_FILE

        np.savez_compressed(data_path, embeddings=embeddings)

        games_csv = Path(__file__).resolve().parents[2] / "data" / "processed" / "games.csv"
        csv_mtime = games_csv.stat().st_mtime_ns if games_csv.exists() else 0

        meta = {
            "provider_id": self.provider.provider_id,
            "dimension": self.provider.dimension,
            "n_games": len(self.games),
            "game_ids_hash": self._game_ids_hash(),
            "corpus_version": SEMANTIC_CORPUS_VERSION,
            "games_csv_mtime_ns": csv_mtime,
        }

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        logger.info("SemanticRecommender: cache salvo em %s.", data_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_scores(self) -> pd.Series:
        """Retorna Series de zeros com o tamanho do catalogo."""
        return pd.Series(
            np.zeros(len(self.games)),
            index=self.games.index,
            name="semantic_score",
        )

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "games"):
            raise RuntimeError("SemanticRecommender precisa ser treinado com .fit(games).")
