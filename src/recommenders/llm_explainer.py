"""Gerador de justificativas usando LLMs via Amazon Bedrock."""

import json
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


class LLMExplainer:
    """Gera justificativas textuais usando Claude 3 no Amazon Bedrock."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.is_active = False

        if not _has_aws_configuration():
            logger.info("LLMExplainer desabilitado: credenciais AWS nao configuradas.")
            return
        
        try:
            import boto3
            
            self._client = boto3.client("bedrock-runtime")
            # Tentativa de inicializacao passiva. Nao faremos teste ativo para economizar custo.
            # Se falhar durante a geracao, trataremos o erro.
            self.is_active = True
            logger.info("LLMExplainer inicializado (%s)", self.model_id)
        except Exception as e:
            logger.warning("LLMExplainer desabilitado (sem boto3 ou AWS): %s", e)

    def generate_explanation(
        self,
        game_row: pd.Series,
        query: str,
        reference_label: str,
        score: float,
    ) -> str:
        """Gera a explicacao via Claude 3."""
        if not self.is_active:
            return "O motor de IA Generativa (Amazon Bedrock) está offline. Configure suas credenciais AWS para ativar justificativas dinâmicas."

        import botocore.exceptions
        
        prompt = self._build_prompt(game_row, query, reference_label, score)
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content", [{}])[0].get("text", "").strip()
        except botocore.exceptions.ClientError as e:
            logger.error("Erro no Bedrock LLM: %s", e)
            return "Falha ao contactar a IA. Verifique as cotas da conta AWS."

    def _build_prompt(
        self,
        game_row: pd.Series,
        query: str,
        reference_label: str,
        score: float,
    ) -> str:
        """Monta o prompt para o Claude."""
        title = game_row.get("title", "Desconhecido")
        genres = game_row.get("genres", "")
        tags = game_row.get("tags", "")
        desc = game_row.get("description", "")
        
        context_parts = []
        if query:
            context_parts.append(f"o usuário buscou por: '{query}'")
        if reference_label and reference_label != "Nenhum":
            context_parts.append(f"o usuário gosta de: {reference_label}")
            
        user_context = " e ".join(context_parts) if context_parts else "o usuário pediu uma recomendação geral"
        
        prompt = f"""Você é o recomendador inteligente do Ludex.
Explique em no máximo 3 frases (direto ao ponto, sem saudações) por que o jogo "{title}" é uma ótima recomendação sabendo que {user_context}.

Dados do jogo:
Gêneros: {genres}
Tags: {tags}
Descrição: {desc}
Score de Relevância: {score:.2f}

Evite jargões matemáticos (como "score de relevância"). Foque nos elementos do jogo (mecânicas, estilo, gênero) que casam com o que o usuário quer.
Justificativa:"""
        return prompt


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
