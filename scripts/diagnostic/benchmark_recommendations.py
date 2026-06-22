from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_games
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender, HybridWeights
from src.recommenders.ollama_explainer import OllamaExplainer


BENCHMARK_CASES = [
    {"name": "Busca: roguelike desafiador", "query": "roguelike desafiador", "references": []},
    {"name": "Busca: fazenda relaxante", "query": "jogo de fazenda relaxante", "references": []},
    {"name": "Busca: cyberpunk neon", "query": "cyberpunk neon", "references": []},
    {"name": "Busca: RPG narrativo", "query": "rpg narrativo com escolhas", "references": []},
    {"name": "Referencia: Minecraft Dungeons", "query": "", "references": ["Minecraft Dungeons", "Minecraft: Dungeons"]},
    {"name": "Referencia: Hades", "query": "", "references": ["Hades"]},
    {"name": "Referencia: Stardew Valley", "query": "", "references": ["Stardew Valley"]},
    {"name": "Referencia: Disco Elysium", "query": "", "references": ["Disco Elysium"]},
]

ASSISTANT_CASES = [
    "roguelike desafiador",
    "jogo de fazenda relaxante",
    "cyberpunk neon",
    "rpg narrativo com escolhas",
]

# Pesos quando o motor semantico esta ativo.
SEMANTIC_WEIGHTS = HybridWeights()


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


def resolve_reference_ids(games: pd.DataFrame, candidates: list[str]) -> list[str]:
    titles = games["title"].fillna("").astype(str)
    normalized_titles = titles.map(normalize)

    for candidate in candidates:
        target = normalize(candidate)
        exact_matches = games.loc[normalized_titles == target, "game_id"]
        if not exact_matches.empty:
            return [str(exact_matches.iloc[0])]

        contains_matches = games.loc[normalized_titles.str.contains(target, regex=False), "game_id"]
        if not contains_matches.empty:
            return [str(contains_matches.iloc[0])]

    return []


def build_explanation(recommendation: pd.Series) -> str:
    parts = []
    if float(recommendation.get("content_score", 0.0)) > 0:
        parts.append(f"conteudo={float(recommendation['content_score']):.3f}")
    if float(recommendation.get("semantic_score", 0.0)) > 0:
        parts.append(f"semantico={float(recommendation['semantic_score']):.3f}")
    if float(recommendation.get("opinion_score", 0.0)) > 0:
        parts.append(f"busca={float(recommendation['opinion_score']):.3f}")
    parts.append(f"qualidade={float(recommendation.get('quality_score', 0.0)):.3f}")
    return ", ".join(parts)


def format_results(results: pd.DataFrame) -> str:
    columns = ["title", "score", "content_score", "semantic_score", "opinion_score", "quality_score", "positive_ratio"]
    existing_columns = [column for column in columns if column in results.columns]
    view = results[existing_columns].copy()
    for column in ["score", "content_score", "semantic_score", "opinion_score", "quality_score"]:
        if column in view.columns:
            view[column] = view[column].map(lambda value: f"{float(value):.4f}")
    if "positive_ratio" in view.columns:
        view["positive_ratio"] = view["positive_ratio"].map(lambda value: f"{float(value):.0f}%")
    view["explanation"] = results.apply(build_explanation, axis=1)
    return view.to_string(index=False)


def format_assistant_context(results: pd.DataFrame) -> str:
    columns = ["title", "score", "genres", "tags", "positive_ratio"]
    existing_columns = [column for column in columns if column in results.columns]
    view = results[existing_columns].copy()
    if "score" in view.columns:
        view["score"] = view["score"].map(lambda value: f"{float(value):.4f}")
    if "positive_ratio" in view.columns:
        view["positive_ratio"] = view["positive_ratio"].map(lambda value: f"{float(value):.0f}%")
    return view.to_string(index=False)


def try_build_semantic_recommender(games: pd.DataFrame):
    """Tenta criar SemanticRecommender com provider local. Retorna None se falhar."""
    try:
        from src.recommenders.embeddings import create_embedding_provider
        from src.recommenders.semantic import SemanticRecommender

        provider = create_embedding_provider()
        if provider is None:
            return None

        recommender = SemanticRecommender(provider=provider)
        recommender.fit(games)
        return recommender if recommender.is_active else None
    except Exception as exc:
        print(f"⚠️  Semantic recommender indisponivel: {exc}")
        return None


def run_single_benchmark(
    label: str,
    hybrid: HybridRecommender,
    games: pd.DataFrame,
    top_n: int,
    emit,
) -> None:
    """Roda todos os cenarios de benchmark para um HybridRecommender."""
    emit(f"\n{'=' * 72}")
    emit(f"  {label}")
    emit(f"{'=' * 72}")

    for case in BENCHMARK_CASES:
        reference_ids = resolve_reference_ids(games, case["references"])
        case_start = time.perf_counter()
        recommendations = hybrid.recommend(
            games=games,
            query=case["query"],
            reference_game_ids=reference_ids or None,
            top_n=top_n,
        )
        case_duration = time.perf_counter() - case_start

        emit()
        emit(f"## {case['name']}")
        if case["query"]:
            emit(f"Query: {case['query']}")
        if case["references"]:
            status = ", ".join(reference_ids) if reference_ids else "NAO ENCONTRADA"
            emit(f"Referencia resolvida: {status}")
        emit(f"Tempo de ranking: {case_duration:.3f}s")
        emit(format_results(recommendations))


def run_assistant_benchmark(
    hybrid: HybridRecommender,
    games: pd.DataFrame,
    top_n: int,
    emit,
) -> None:
    """Registra o caminho RAG do Assistente e o fallback offline."""
    explainer = OllamaExplainer()
    explainer.is_active = False
    if not explainer.status_message:
        explainer.status_message = "Fallback offline forcado para benchmark."

    emit(f"\n{'=' * 72}")
    emit("  ASSISTENTE LUDEX: RAG + fallback offline")
    emit(f"{'=' * 72}")
    emit(f"Modelo esperado: {explainer.model_id}")
    emit(f"Endpoint configurado: {explainer.endpoint_url}")

    for query in ASSISTANT_CASES:
        case_start = time.perf_counter()
        context = explainer.retrieve_context(
            user_prompt=query,
            history=[],
            games_df=games,
            recommender=hybrid,
            top_n=top_n,
        )
        fallback_response = explainer.chat(query, [], games, hybrid)
        case_duration = time.perf_counter() - case_start

        emit()
        emit(f"## Assistente: {query}")
        emit(f"Query: {query}")
        emit(f"Tempo RAG + fallback: {case_duration:.3f}s")
        emit("Top recuperado pelo RAG:")
        emit(format_assistant_context(context))
        emit("Resposta fallback offline:")
        emit(fallback_response)


def run_benchmark(top_n: int, output_path: str | None = None) -> None:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    load_start = time.perf_counter()
    games = load_games()
    load_duration = time.perf_counter() - load_start

    fit_start = time.perf_counter()
    content = ContentBasedRecommender().fit(games)
    fit_duration = time.perf_counter() - fit_start

    emit(f"Catalogo: {len(games)} jogos")
    emit(f"Tempo de carga: {load_duration:.2f}s")
    emit(f"Tempo de fit TF-IDF: {fit_duration:.2f}s")

    # --- Pass 1: TF-IDF only (baseline) ---
    hybrid_tfidf = HybridRecommender(content_recommender=content)
    run_single_benchmark(
        label="MOTOR: TF-IDF Only (baseline)",
        hybrid=hybrid_tfidf,
        games=games,
        top_n=top_n,
        emit=emit,
    )

    # --- Pass 2: Hybrid + Semantic (se disponivel) ---
    emit("")
    emit("Tentando inicializar motor semantico...")
    semantic_start = time.perf_counter()
    semantic = try_build_semantic_recommender(games)
    semantic_duration = time.perf_counter() - semantic_start

    if semantic is not None:
        emit(f"Motor semantico ativo: {semantic.provider.provider_id} ({semantic_duration:.2f}s)")
        hybrid_semantic = HybridRecommender(
            content_recommender=content,
            semantic_recommender=semantic,
            weights=SEMANTIC_WEIGHTS,
        )
        run_single_benchmark(
            label=f"MOTOR: Hibrido TF-IDF + Semantico ({SEMANTIC_WEIGHTS})",
            hybrid=hybrid_semantic,
            games=games,
            top_n=top_n,
            emit=emit,
        )
    else:
        emit(
            "Motor semantico indisponivel. Rode com LUDEX_EMBEDDING_PROVIDER=local "
            "e modelo local completo para comparar; Bedrock permanece legado/opcional."
        )

    run_assistant_benchmark(
        hybrid=hybrid_tfidf,
        games=games,
        top_n=top_n,
        emit=emit,
    )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nResultados salvos em: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local de recomendacoes Ludex.")
    parser.add_argument("--top-n", type=int, default=5, help="Quantidade de recomendacoes por caso.")
    parser.add_argument("--output", type=str, default=None, help="Salvar saida em arquivo.")
    args = parser.parse_args()
    run_benchmark(top_n=args.top_n, output_path=args.output)


if __name__ == "__main__":
    main()
