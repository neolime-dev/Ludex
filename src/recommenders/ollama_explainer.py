"""Gerador de justificativas e ChatBot usando Ollama local."""

import logging
import os
import re
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LUDEX_OLLAMA_MODEL", "ludex-assistant:latest")
DEFAULT_ENDPOINT_URL = os.getenv("LUDEX_OLLAMA_URL", "http://localhost:11434")
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 900
MAX_CONTEXT_GAMES = 5
MAX_CONTEXT_FIELD_CHARS = 260
ENGLISH_STOPWORDS = {
    "the",
    "and",
    "as",
    "with",
    "from",
    "for",
    "that",
    "this",
    "your",
    "you",
    "are",
    "have",
    "play",
    "highly",
    "trained",
    "early",
    "childhood",
    "mutated",
    "superhuman",
    "world",
    "game",
    "skills",
    "strength",
    "story",
    "monster",
    "hire",
}


class OllamaExplainer:
    """Gera justificativas e conversas iterativas usando um LLM local."""

    def __init__(self, model_id: str | None = None, endpoint_url: str | None = None) -> None:
        self.model_id = model_id or DEFAULT_MODEL
        self.endpoint_url = self._normalize_endpoint(endpoint_url or DEFAULT_ENDPOINT_URL)
        self.status_message = ""
        self.is_active = self._check_health()
        if self.is_active:
            logger.info("OllamaExplainer conectado com sucesso ao modelo: %s", self.model_id)
        else:
            logger.warning("OllamaExplainer inativo: %s", self.status_message)

    def _check_health(self) -> bool:
        """Verifica se o servidor responde e se o modelo esperado existe."""
        try:
            response = requests.get(f"{self.endpoint_url}/api/tags", timeout=2)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Falha no health check do Ollama em %s: %s", self.endpoint_url, exc)
            self.status_message = (
                f"Ollama indisponivel em {self.endpoint_url}. "
                "Inicie o servidor local e confirme se o modelo esperado aparece em /api/tags."
            )
            return False

        models = payload.get("models", [])
        available = {
            str(model.get("name") or model.get("model") or "")
            for model in models
            if isinstance(model, dict)
        }
        expected = self.model_id
        expected_with_tag = expected if ":" in expected else f"{expected}:latest"
        if expected not in available and expected_with_tag not in available:
            self.status_message = (
                f"Modelo '{self.model_id}' nao encontrado no Ollama. "
                f"Crie com: ollama create ludex-assistant -f ludex-llama3-finetuned_gguf/Modelfile"
            )
            return False

        self.status_message = f"Modelo '{self.model_id}' ativo em {self.endpoint_url}."
        return True

    @staticmethod
    def _normalize_endpoint(endpoint_url: str) -> str:
        endpoint = endpoint_url.strip().rstrip("/")
        if endpoint.endswith("/api"):
            endpoint = endpoint[:-4]
        return endpoint

    def generate_explanation(
        self,
        game_row: pd.Series,
        query: str,
        reference_label: str,
        score: float,
    ) -> str:
        """Gera a explicacao de recomendacao (3 frases)."""
        if not self.is_active:
            return self._local_explanation(game_row, query, reference_label)

        prompt = self._build_prompt(game_row, query, reference_label, score)

        try:
            r = requests.post(
                f"{self.endpoint_url}/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=20,
            )
            r.raise_for_status()
            response = r.json().get("response", "").strip()
            return self._clean_model_response(response) or self._local_explanation(game_row, query, reference_label)
        except Exception as e:
            logger.error("Erro no Ollama Generate: %s", e)
            return self._local_explanation(game_row, query, reference_label)

    def chat(self, user_prompt: str, history: list[dict], games_df: pd.DataFrame, recommender: Any) -> str:
        """Modo conversacional (ChatBot) com RAG Autônomo via Motor Híbrido."""
        try:
            context_games = self.retrieve_context(user_prompt, history, games_df, recommender)
        except Exception as e:
            logger.error("Erro no recommender durante chat: %s", e)
            return "Ops! Ocorreu um erro ao consultar o catálogo do Ludex."

        if not self.is_active:
            return self._local_chat_response(user_prompt, context_games)

        context_text = self._format_context(context_games)
        system_msg = {
            "role": "system",
            "content": (
                "Você é o 'Assistente Ludex', um expert recomendador de jogos. "
                "Você acaba de fazer uma busca secreta no banco de dados baseada no pedido do usuário e encontrou esses resultados no catálogo:\n"
                "REGRAS ESTRITAS:\n"
                "1. Baseie sua resposta ESTRITAMENTE nos JOGOS NO CONTEXTO abaixo. Nunca alucine, minta ou tire do nada o nome de um jogo que não consta no contexto abaixo.\n"
                "2. Explique ao usuário de forma amigável por que o jogo escolhido da lista combina com a intenção dele, baseando-se nos Gêneros e Tags.\n"
                "3. Responda INTEIRAMENTE em PORTUGUÊS DO BRASIL.\n"
                "4. Não copie descrições, sinopses ou frases promocionais do catálogo. Use somente gêneros e tags como sinais.\n"
                "5. Seja direto: recomende no máximo 3 jogos e explique em uma frase curta cada um.\n\n"
                f"JOGOS NO CONTEXTO DA SUA BUSCA:\n{context_text}"
            ),
        }

        clean_history = self._clean_history(history)
        api_messages = [system_msg] + clean_history + [{"role": "user", "content": user_prompt}]

        try:
            r = requests.post(
                f"{self.endpoint_url}/api/chat",
                json={
                    "model": self.model_id,
                    "messages": api_messages,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=45,
            )
            r.raise_for_status()
            response = r.json().get("message", {}).get("content", "").strip()
            return self._clean_model_response(response) or self._local_chat_response(user_prompt, context_games)
        except Exception as e:
            logger.error("Erro no Ollama Chat: %s", e)
            return self._local_chat_response(user_prompt, context_games)

    def retrieve_context(
        self,
        user_prompt: str,
        history: list[dict],
        games_df: pd.DataFrame,
        recommender: Any,
        top_n: int = MAX_CONTEXT_GAMES,
    ) -> pd.DataFrame:
        """Recupera o top-N usado como contexto do assistente."""
        history_user_msgs = [
            str(message.get("content", ""))
            for message in history
            if message.get("role") == "user"
        ]
        recent_history = " ".join(history_user_msgs[-2:])
        search_query = f"{recent_history} {user_prompt}".strip()
        return recommender.recommend(games_df, query=search_query, top_n=top_n).head(top_n)

    def _build_prompt(
        self,
        game_row: pd.Series,
        query: str,
        reference_label: str,
        score: float,
    ) -> str:
        """Monta o prompt de justificativa."""
        title = game_row.get("title", "Desconhecido")
        genres = game_row.get("genres", "")
        tags = game_row.get("tags", "")

        context_parts = []
        if query:
            context_parts.append(f"o usuário buscou por: '{query}'")
        if reference_label and reference_label != "Nenhum":
            context_parts.append(f"o usuário gosta de: {reference_label}")

        user_context = " e ".join(context_parts) if context_parts else "o usuário pediu uma recomendação geral"

        return f"""Responda em PORTUGUÊS. Você é o recomendador inteligente do Ludex.
Explique em no máximo 3 frases diretas por que o jogo "{title}" é uma ótima recomendação sabendo que {user_context}.

Dados do jogo:
Gêneros: {genres}
Tags: {tags}

Regras:
- Responda apenas em português do Brasil.
- Não copie sinopse, descrição ou texto promocional do catálogo.
- Foque nas mecânicas e estilo sugeridos por gêneros e tags.
- Não use jargões matemáticos e não inicie com saudações.
Justificativa:"""

    def _local_explanation(self, game_row: pd.Series, query: str, reference_label: str) -> str:
        title = game_row.get("title", "este jogo")
        genres = self._short_field(game_row.get("genres", ""))
        tags = self._short_field(game_row.get("tags", ""))
        intent = query or (f"um perfil parecido com {reference_label}" if reference_label != "Nenhum" else "uma boa recomendação")
        return (
            f"Modo local ativo: {title} combina com {intent} pelos gêneros {genres or 'do catálogo'} "
            f"e pelas tags {tags or 'associadas ao jogo'}. A ordem vem do ranking híbrido do Ludex, "
            "usando TF-IDF, busca textual e qualidade da comunidade."
        )

    def _local_chat_response(self, user_prompt: str, context_games: pd.DataFrame) -> str:
        if context_games.empty:
            return (
                "Estou em modo local e não encontrei bons candidatos com esse pedido. "
                "Tente descrever gênero, mecânica ou clima do jogo com outras palavras."
            )

        lines = [
            "Estou em modo local, então usei o ranking do Ludex sem acionar o Ollama.",
            "Minhas melhores sugestões são:",
        ]
        for position, (_, row) in enumerate(context_games.head(MAX_CONTEXT_GAMES).iterrows(), start=1):
            title = row.get("title", "Jogo sem titulo")
            genres = self._short_field(row.get("genres", ""))
            tags = self._short_field(row.get("tags", ""))
            reason = tags or genres or "sinais do catálogo"
            lines.append(f"{position}. {title} - combina com '{user_prompt}' por {reason}.")
        lines.append(
            f"Para ativar a resposta com LLM, deixe o Ollama rodando com o modelo {self.model_id}. "
            f"{self.status_message}"
        )
        return "\n".join(lines)

    def _format_context(self, context_games: pd.DataFrame) -> str:
        lines = []
        for _, row in context_games.head(MAX_CONTEXT_GAMES).iterrows():
            title = self._short_field(row.get("title", ""))
            genres = self._short_field(row.get("genres", ""))
            tags = self._short_field(row.get("tags", ""))
            lines.append(
                f"- Título: {title} | Gêneros: {genres} | Tags: {tags}"
            )
        return "\n".join(lines)

    def _clean_history(self, history: list[dict]) -> list[dict]:
        cleaned = []
        for message in history[-MAX_HISTORY_MESSAGES:]:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = self._short_field(message.get("content", ""), MAX_HISTORY_CHARS)
            cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _short_field(value: object, limit: int = MAX_CONTEXT_FIELD_CHARS) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    def _clean_model_response(self, response: str) -> str:
        """Remove trechos em ingles que o modelo possa copiar do dataset antigo."""
        text = " ".join(str(response or "").split())
        if not text:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", text)
        kept = [part for part in parts if not self._looks_like_english_sentence(part)]
        cleaned = " ".join(kept).strip()
        removed_english = len(kept) < len(parts)
        if len(cleaned) < 24 and len(text) > len(cleaned):
            return ""
        if removed_english and not self._has_recommendation_reason(cleaned):
            return ""
        return cleaned

    @staticmethod
    def _has_recommendation_reason(text: str) -> bool:
        return bool(
            re.search(
                r"\b(combina|porque|por conta|gênero|genero|tags?|mecânica|mecanica|estilo|experiência|experiencia|foco|destaque)\b",
                text.lower(),
            )
        )

    @staticmethod
    def _looks_like_english_sentence(text: str) -> bool:
        words = re.findall(r"[A-Za-z']+", text.lower())
        if len(words) < 7:
            return False

        english_hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
        portuguese_markers = len(
            re.findall(
                r"\b(que|com|para|por|porque|jogo|jogos|voce|você|recomendo|combina|generos|gêneros|tags|busca)\b",
                text.lower(),
            )
        )
        return english_hits >= 3 and english_hits > portuguese_markers
