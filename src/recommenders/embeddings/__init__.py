"""Pacote de provedores de embeddings para o Ludex.

A factory e intencionalmente conservadora para nao travar a demo:
por padrao, o provider local so usa arquivos ja presentes no cache do
Hugging Face. Downloads precisam ser habilitados explicitamente com
``LUDEX_ALLOW_MODEL_DOWNLOAD=1``.
"""

from __future__ import annotations

import logging
import os

from src.recommenders.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


def create_embedding_provider() -> EmbeddingProvider | None:
    """Tenta instanciar o melhor provider disponivel.

    ``LUDEX_EMBEDDING_PROVIDER`` aceita ``auto`` (padrao), ``local``,
    ``bedrock`` ou ``off``. Em ``auto``, Bedrock so e tentado quando ha
    indicio de configuracao AWS. O provider local e opt-in porque a stack
    do Hugging Face pode tentar rede quando o cache esta incompleto.
    """
    selected = os.getenv("LUDEX_EMBEDDING_PROVIDER", "auto").strip().lower()
    if selected in {"off", "none", "disabled", "0"}:
        return None

    if selected in {"bedrock", "auto"} and _has_aws_configuration():
        provider = _try_bedrock_provider()
        if provider is not None or selected == "bedrock":
            return provider

    if selected == "local":
        provider = _try_local_provider()
        return provider

    if selected == "bedrock":
        return _try_bedrock_provider()

    return None


def _try_local_provider() -> EmbeddingProvider | None:
    try:
        from src.recommenders.embeddings.local_provider import LocalEmbeddingProvider

        return LocalEmbeddingProvider()
    except Exception as exc:
        logger.warning("Embeddings locais indisponiveis: %s", exc)
        return None


def _try_bedrock_provider() -> EmbeddingProvider | None:
    try:
        from src.recommenders.embeddings.bedrock_provider import BedrockEmbeddingProvider

        return BedrockEmbeddingProvider()
    except Exception as exc:
        logger.warning("Embeddings Bedrock indisponiveis: %s", exc)
        return None


def _has_aws_configuration() -> bool:
    return any(
        os.getenv(name)
        for name in [
            "AWS_ACCESS_KEY_ID",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        ]
    )


__all__ = ["EmbeddingProvider", "create_embedding_provider"]
