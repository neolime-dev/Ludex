"""Provider de embeddings local usando sentence-transformers.

Usa o modelo ``all-MiniLM-L6-v2`` (~80MB, 384 dimensoes) que roda
em CPU ou GPU sem depender de servicos externos. Os vetores sao
normalizados em L2, permitindo que o produto escalar seja usado
diretamente como similaridade do cosseno.

Requisito: ``pip install sentence-transformers``
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.recommenders.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Modelo de alta capacidade (~420MB download, 768 dims).
# Melhor qualidade semantica disponivel para modelos under-1GB,
# ideal para composicao de conceitos (ex: ARPG + Voxel).
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


class LocalEmbeddingProvider(EmbeddingProvider):
    """Gera embeddings locais com sentence-transformers.

    Se CUDA estiver disponivel, usa GPU automaticamente.
    Caso contrario, roda em CPU (mais lento, mas funcional).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, local_files_only: bool | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        if local_files_only is None:
            local_files_only = os.getenv("LUDEX_ALLOW_MODEL_DOWNLOAD", "").lower() not in {"1", "true", "yes"}
        self._local_files_only = local_files_only
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
        device = str(self._model.device)
        logger.info(
            "LocalEmbeddingProvider inicializado: modelo=%s, device=%s, local_files_only=%s",
            model_name,
            device,
            local_files_only,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos.

        Usa batching interno de 128 textos e mostra barra de progresso.
        Os vetores sao L2-normalizados pelo sentence-transformers.
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Gera embedding para uma unica query de busca."""
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    @property
    def provider_id(self) -> str:
        return f"local:{self._model_name}"
