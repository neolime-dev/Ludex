"""Provider de embeddings usando Amazon Bedrock (Titan V2).

Requer credenciais AWS configuradas (via ~/.aws/credentials ou variaveis de ambiente).
Gera embeddings de 1024 dimensoes, de alta qualidade.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.recommenders.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Gera embeddings via Amazon Bedrock Runtime.

    Requer boto3 e acesso a API do Bedrock.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        self._model_id = model_id
        
        # Tenta inicializar o cliente. Se falhar por falta de credenciais, repassa o erro
        try:
            self._client = boto3.client("bedrock-runtime")
            # Faz uma chamada de teste rapida para validar acesso
            self.embed_query("test")
            logger.info("BedrockEmbeddingProvider inicializado com sucesso (%s).", model_id)
        except (BotoCoreError, ClientError) as e:
            logger.warning("Falha ao inicializar Bedrock (sem credenciais ou acesso): %s", e)
            raise RuntimeError(f"Bedrock nao disponivel: {e}") from e

    def _embed_single(self, text: str) -> list[float]:
        """Chama a API do Titan V2 para um unico texto."""
        import botocore.exceptions
        import time

        body = json.dumps({
            "inputText": text,
            "dimensions": DIMENSIONS,
            "normalize": True
        })
        
        # Retentativa simples para throttling
        for attempt in range(5):
            try:
                response = self._client.invoke_model(
                    body=body,
                    modelId=self._model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                response_body = json.loads(response.get("body").read())
                return response_body.get("embedding", [])
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "ThrottlingException":
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Falha de throttling persistente no Bedrock")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos em paralelo."""
        embeddings = [None] * len(texts)
        
        # Usa ThreadPool para paralelizar requisicoes a API
        # Limita workers para nao estourar o limite de TPS da AWS rapidamente
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(self._embed_single, text): idx 
                for idx, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    logger.error("Erro ao gerar embedding para indice %d: %s", idx, e)
                    # Fallback para vetor zero se um texto falhar criticamente
                    embeddings[idx] = [0.0] * DIMENSIONS

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Gera embedding para uma unica query."""
        embedding = self._embed_single(query)
        return np.asarray([embedding], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return DIMENSIONS

    @property
    def provider_id(self) -> str:
        return f"bedrock:{self._model_id}"
