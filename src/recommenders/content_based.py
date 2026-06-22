from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEXT_COLUMNS = ["title", "genres", "tags", "description"]
OPINION_COLUMNS = ["review_keywords"]
RECOMMENDER_MODEL_VERSION = 7

QUERY_EXPANSIONS = {
    "acao": "action fast-paced combat",
    "aventura": "adventure exploration",
    "cartas": "card deckbuilding",
    "cooperativo": "co-op cooperative multiplayer",
    "criacao": "crafting building",
    "desafiador": "difficult challenging hard",
    "desafiadores": "difficult challenging hard",
    "detetive": "detective mystery murder case",
    "dificil": "difficult challenging hard",
    "divertido": "fun funny enjoyable comedy",
    "emocionante": "emotional moving touching story rich",
    "emocional": "emotional moving touching story rich",
    "enjoativo": "boring repetitive tedious",
    "envolvente": "immersive engaging atmospheric",
    "escolha": "choices matter multiple endings decision dialogue",
    "escolhas": "choices matter multiple endings decision dialogue",
    "fantasia": "fantasy",
    "fazenda": "farming agriculture cozy farming life sim relaxing",
    "frustrante": "frustrating difficult unfair rage",
    "historia": "story rich narrative choices matter",
    "humor": "funny comedy",
    "imersiva": "immersive atmospheric deep",
    "imersivo": "immersive atmospheric deep",
    "investigacao": "detective mystery murder case",
    "mitologia": "mythology fantasy",
    "mundo": "world open world",
    "narrativa": "story rich narrative choices matter",
    "narrativo": "story rich narrative choices matter dialogue",
    "nostalgico": "nostalgic retro classic old school",
    "obra": "masterpiece great excellent",
    "quebra": "puzzle",
    "rapido": "fast-paced action",
    "recompensador": "rewarding satisfying progression",
    "relaxante": "relaxing casual cozy cozy farming life sim",
    "repetitivo": "repetitive grind tedious",
    "roguelike": "roguelike roguelite challenging roguelike",
    "rpg": "rpg role-playing",
    "rpgs": "rpg role-playing",
    "satisfatorio": "satisfying rewarding",
    "simulacao": "simulation sim",
    "sombria": "dark dark fantasy",
    "sombrio": "dark dark fantasy",
    "tatica": "tactical strategy turn-based",
    "taticas": "tactical strategy turn-based",
    "tatico": "tactical strategy turn-based",
    "taticos": "tactical strategy turn-based",
    "viciante": "addictive replayable replay value progression",
}

NOISY_TAG_TERMS = {
    "4 giocatori divano",
    "action rpg",
    "captions available",
    "commentary available",
    "exclusive",
    "full controller support",
    "isometric",
    "partial controller support",
    "remote play together",
    "role playing",
    "role playing game",
    "role-playing",
    "rpg",
    "shared split screen",
    "singleplayer",
    "stats",
    "steam achievements",
    "steam cloud",
    "steam leaderboards",
    "steam trading cards",
    "cloud saves",
    "overlay",
    "steam workshop",
    "steam-trading-cards",
    "top down",
    "true exclusive",
}

SEMANTIC_HINT_RULES = {
    "action rpg": [
        r"\baction rpg\b",
        r"\baction role.?playing\b",
        r"\brpg elements\b",
        r"\baction\b.{0,80}\brpg\b",
        r"\brpg\b.{0,80}\baction\b",
    ],
    "choices matter": [r"\bchoices matter\b", r"\bmultiple endings\b", r"\bdifficult choices\b", r"\bopen ended\b"],
    "challenging difficult": [r"\bdifficult\b", r"\bchallenging\b", r"\bhard\b", r"\bpunishing\b"],
    "co-op": [r"\bco-?op\b", r"\bcooperative\b", r"\bteam up\b", r"\bup to four\b"],
    "crafting building": [r"\bcrafting\b", r"\bbuilding\b", r"\bbuild a\b", r"\bsandbox\b"],
    "crpg tabletop": [r"\bcrpg\b", r"\bcomputer rpg\b", r"\btabletop\b", r"\bpen and paper\b"],
    "cyberpunk": [r"\bcyberpunk\b", r"\bneon\b", r"\bdystopian\b"],
    "deckbuilding card": [r"\bdeck.?building\b", r"\bdeck.?builder\b", r"\bcard game\b"],
    "detective mystery": [r"\bdetective\b", r"\bmurder\b", r"\binterrogate\b", r"\bsuspects?\b", r"\bcop\b", r"\bcase\b"],
    "dungeon crawler": [r"\bdungeon crawler\b", r"\bdungeon crawlers\b", r"\bdungeons?\b"],
    "farming agriculture": [r"\bfarm\b", r"\bfarming\b", r"\bagriculture\b", r"\bcrops?\b"],
    "hack and slash": [r"\bhack and slash\b", r"\bhack & slash\b", r"\bslay hordes\b"],
    "isometric top down": [r"\bisometric\b", r"\btop.?down\b"],
    "loot treasure": [r"\bloot\b", r"\blooting\b", r"\btreasure\b", r"\brelics?\b"],
    "metroidvania": [r"\bmetroidvania\b"],
    "narrative story": [r"\bstory rich\b", r"\bnarrative\b", r"\bdialogue\b", r"\bopen ended case\b", r"\bsolve\b"],
    "open world": [r"\bopen world\b", r"\bvast world\b"],
    "platformer": [r"\bplatformer\b", r"\bplatforming\b"],
    "political philosophical": [r"\bpolitical\b", r"\bpolitics\b", r"\bphilosophical\b", r"\bphilosophy\b"],
    "puzzle": [r"\bpuzzle\b", r"\bpuzzles\b"],
    "relaxing cozy": [r"\brelaxing\b", r"\bcozy\b", r"\bcosy\b"],
    "roguelike roguelite": [r"\broguelike\b", r"\broguelite\b", r"\bprocedural\b", r"\brandomized\b"],
    "survival": [r"\bsurvival\b", r"\bsurvive\b"],
    "turn based tactical": [r"\bturn.?based\b", r"\btactical\b", r"\btactics\b"],
}

COMPILED_SEMANTIC_HINT_RULES = {
    label: [re.compile(pattern) for pattern in patterns]
    for label, patterns in SEMANTIC_HINT_RULES.items()
}


@dataclass
class ContentBasedRecommender:
    max_features: int = 30000
    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)

    def fit(self, games: pd.DataFrame) -> "ContentBasedRecommender":
        self.games = games.reset_index(drop=True).copy()
        self.model_cache_version = RECOMMENDER_MODEL_VERSION
        text_features = _prepare_text_features(_safe_text_frame(self.games, TEXT_COLUMNS + OPINION_COLUMNS))
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        self.matrix = self.vectorizer.fit_transform(self._build_corpus_from_features(text_features))
        self.opinion_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
        )
        self.opinion_matrix = self.opinion_vectorizer.fit_transform(self._build_opinion_corpus_from_features(text_features))
        self.game_id_to_index = {
            str(game_id): index for index, game_id in enumerate(self.games["game_id"].astype(str))
        }
        return self

    def score_by_text(self, query: str) -> pd.Series:
        self._ensure_fitted()
        if not str(query or "").strip():
            return self._empty_scores()

        query_vector = self.vectorizer.transform([expand_query(query)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        return pd.Series(scores, index=self.games.index, name="content_score")

    def score_by_opinion_text(self, query: str) -> pd.Series:
        self._ensure_fitted()
        if not str(query or "").strip():
            return self._empty_scores().rename("opinion_score")

        query_vector = self.opinion_vectorizer.transform([expand_query(query)])
        scores = cosine_similarity(query_vector, self.opinion_matrix).ravel()
        return pd.Series(scores, index=self.games.index, name="opinion_score")

    def score_by_game_ids(self, game_ids: list[str]) -> pd.Series:
        self._ensure_fitted()
        valid_indices = [
            self.game_id_to_index.get(str(gid))
            for gid in game_ids
            if self.game_id_to_index.get(str(gid)) is not None
        ]

        if not valid_indices:
            return self._empty_scores()

        # Calcula o vetor medio (centroide) de todos os jogos selecionados.
        centroid_vector = self.matrix[valid_indices].mean(axis=0)

        if isinstance(centroid_vector, np.matrix):
            centroid_vector = np.asarray(centroid_vector)
        elif hasattr(centroid_vector, "toarray"):
            centroid_vector = centroid_vector.toarray()

        scores = cosine_similarity(centroid_vector, self.matrix).ravel()

        # Zera o score dos próprios jogos de referência para não recomendar o que já foi selecionado
        for idx in valid_indices:
            scores[idx] = 0.0

        return pd.Series(scores, index=self.games.index, name="content_score")

    def score_by_game_id(self, game_id: str) -> pd.Series:
        return self.score_by_game_ids([game_id])

    def top_terms_for_recommendation(
        self,
        recommended_game_id: str,
        query: str | None = None,
        reference_game_id: str | None = None,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Retorna as palavras/n-grams que mais explicam uma recomendacao TF-IDF.

        A contribuicao de cada termo para a similaridade do cosseno e o produto
        elemento a elemento entre o vetor TF-IDF do lado da consulta (texto buscado
        ou jogo de referencia) e o vetor do jogo recomendado. Como as linhas do
        TfidfVectorizer ja sao normalizadas em L2, esse produto soma exatamente o
        cosseno usado no ranqueamento, entao os maiores termos sao os "porques".

        Ordem de prioridade do lado da consulta:
        1. ``query`` (busca textual), se informada;
        2. ``reference_game_id`` (jogo de referencia), se informado;
        3. fallback: os termos mais fortes do proprio jogo recomendado.

        Retorna uma lista de ``(termo, contribuicao)`` ordenada do maior para o
        menor, com no maximo ``top_n`` itens (vazia se o jogo nao existir).
        """
        self._ensure_fitted()
        target_index = self.game_id_to_index.get(str(recommended_game_id))
        if target_index is None:
            return []

        game_vector = self.matrix[target_index]

        if query and str(query).strip():
            source_vector = self.vectorizer.transform([expand_query(query)])
        elif reference_game_id is not None:
            reference_index = self.game_id_to_index.get(str(reference_game_id))
            if reference_index is None:
                return []
            source_vector = self.matrix[reference_index]
        else:
            # Sem busca nem referencia: destaca os termos mais fortes do proprio jogo.
            source_vector = game_vector

        contributions = game_vector.multiply(source_vector).tocoo()
        feature_names = self.vectorizer.get_feature_names_out()
        ranked = sorted(
            zip(contributions.col, contributions.data),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(str(feature_names[col]), float(value)) for col, value in ranked[:top_n] if value > 0]

    def recommend_by_text(self, query: str, top_n: int = 10) -> pd.DataFrame:
        return self._rank(self.score_by_text(query), top_n=top_n)

    def recommend_similar_games(self, game_id: str, top_n: int = 10) -> pd.DataFrame:
        return self._rank(self.score_by_game_id(game_id), top_n=top_n)

    def _rank(self, scores: pd.Series, top_n: int) -> pd.DataFrame:
        results = self.games.copy()
        results["content_score"] = scores.to_numpy()
        return results.sort_values("content_score", ascending=False).head(top_n)

    def empty_scores(self, name: str = "content_score") -> pd.Series:
        return self._empty_scores().rename(name)

    def _empty_scores(self) -> pd.Series:
        return pd.Series(np.zeros(len(self.games)), index=self.games.index, name="content_score")

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "matrix") or not hasattr(self, "opinion_matrix"):
            raise RuntimeError("ContentBasedRecommender precisa ser treinado com .fit(games).")

    @staticmethod
    def _build_corpus(games: pd.DataFrame) -> pd.Series:
        text = _safe_text_frame(games, TEXT_COLUMNS)
        features = _prepare_text_features(text)
        return ContentBasedRecommender._build_corpus_from_features(features)

    @staticmethod
    def _build_corpus_from_features(features: pd.DataFrame) -> pd.Series:
        # Mecanicas dominam, mas tags operacionais da Steam/plataforma sao removidas.
        return (
            features["title"]
            + " "
            + (features["genres"] + " ") * 5
            + (features["tags_clean"] + " ") * 9
            + (features["semantic_hints"] + " ") * 8
            + features["description"]
        )

    @staticmethod
    def _build_opinion_corpus(games: pd.DataFrame) -> pd.Series:
        text = _safe_text_frame(games, TEXT_COLUMNS + OPINION_COLUMNS)
        features = _prepare_text_features(text)
        return ContentBasedRecommender._build_opinion_corpus_from_features(features)

    @staticmethod
    def _build_opinion_corpus_from_features(features: pd.DataFrame) -> pd.Series:
        return (
            features["title"]
            + " "
            + (features["genres"] + " ") * 2
            + (features["tags_clean"] + " ") * 2
            + (features["semantic_hints"] + " ") * 3
            + features["description"]
            + " "
            + (features["review_keywords"] + " ") * 5
        )


def _safe_text_frame(games: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        if column in games.columns:
            data[column] = games[column].fillna("").astype(str)
        else:
            data[column] = pd.Series([""] * len(games), index=games.index)
    return pd.DataFrame(data, index=games.index)


def _prepare_text_features(text: pd.DataFrame) -> pd.DataFrame:
    features = text.copy()
    features["tags_clean"] = features["tags"].map(_clean_tag_text)
    features["semantic_hints"] = features.apply(_infer_semantic_hints, axis=1)
    return features


def _clean_tag_text(value: str) -> str:
    terms = []
    for raw_term in re.split(r"[,|;/]", str(value or "")):
        term = raw_term.strip()
        if not term:
            continue
        if _normalized_tag_term(term) in NOISY_TAG_TERMS:
            continue
        terms.append(term)
    return " ".join(terms)


def _infer_semantic_hints(row: pd.Series) -> str:
    source = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("genres", "")),
            str(row.get("tags", "")),
            str(row.get("description", "")),
        ]
    )
    normalized = normalize_ascii(source).lower()
    hints = []
    for label, patterns in COMPILED_SEMANTIC_HINT_RULES.items():
        if any(pattern.search(normalized) for pattern in patterns):
            hints.append(label)
    hints.extend(_combo_semantic_hints(hints))
    return " ".join(hints)


def _combo_semantic_hints(hints: list[str]) -> list[str]:
    hint_set = set(hints)
    combos: list[str] = []
    if "action rpg" in hint_set and ({"dungeon crawler", "loot treasure"} & hint_set):
        combos.extend(
            [
                "looter arpg",
                "diablo like",
                "dungeon action rpg",
                "hack and slash action",
            ]
        )
    if "farming agriculture" in hint_set and "relaxing cozy" in hint_set:
        combos.extend(["cozy farming", "farming life sim", "relaxing farm sim"])
    if "choices matter" in hint_set and "narrative story" in hint_set:
        combos.extend(["narrative choices", "story rich choices"])
    if "detective mystery" in hint_set and "narrative story" in hint_set:
        combos.extend(["detective narrative", "investigative dialogue", "mystery choices"])
    if "crpg tabletop" in hint_set and "narrative story" in hint_set:
        combos.extend(["crpg narrative", "tabletop role playing", "dialogue heavy rpg"])
    if "political philosophical" in hint_set and "narrative story" in hint_set:
        combos.extend(["political narrative", "philosophical story"])
    if "roguelike roguelite" in hint_set and "challenging difficult" in hint_set:
        combos.extend(["challenging roguelike", "hard roguelite"])
    return combos


def _normalized_tag_term(value: str) -> str:
    normalized = normalize_ascii(value).lower()
    normalized = re.sub(r"[-_/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def expand_query(query: str) -> str:
    normalized = normalize_ascii(query)
    tokens = re.findall(r"[a-zA-Z0-9]+", normalized.lower())
    expansions = []
    for token in tokens:
        expansions.append(token)
        if token.endswith("s") and len(token) > 3:
            expansions.append(token[:-1])
        mapped = QUERY_EXPANSIONS.get(token)
        if mapped:
            expansions.append(mapped)
    return " ".join(expansions)


def normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in normalized if not unicodedata.combining(character))
