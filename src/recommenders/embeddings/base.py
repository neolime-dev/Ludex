"""Contrato abstrato para provedores de embeddings.

Qualquer provider (local, Bedrock, OpenAI, etc.) deve implementar
esta interface para ser plugavel no SemanticRecommender.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Interface abstrata para geracao de embeddings de texto.

    Implementacoes devem garantir que os vetores retornados sejam
    normalizados em L2 (norma unitaria), permitindo que similaridade
    do cosseno seja calculada como simples produto escalar.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos.

        Args:
            texts: Lista de strings para gerar embeddings.

        Returns:
            Array numpy de shape ``(len(texts), self.dimension)`` com
            vetores normalizados em L2.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Gera embedding para uma unica query de busca.

        Args:
            query: Texto de busca do usuario.

        Returns:
            Array numpy de shape ``(1, self.dimension)`` com vetor
            normalizado em L2.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionalidade dos vetores gerados."""
        ...

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Identificador unico do provider e modelo.

        Usado para invalidacao de cache: se o provider_id mudar,
        o cache de embeddings e regenerado. Formato sugerido:
        ``"local:all-MiniLM-L6-v2"`` ou ``"bedrock:amazon.titan-embed-text-v2:0"``.
        """
        ...
