"""Warmup do cache de embeddings semanticos.

Gera embeddings para todo o catalogo e persiste em
``data/processed/cache/embeddings_catalog.npz`` com metadados
de invalidacao em ``embeddings_meta.json``.

Uso:
    LUDEX_EMBEDDING_PROVIDER=local LUDEX_ALLOW_MODEL_DOWNLOAD=1 python scripts/warmup_embeddings.py

Requisitos:
    - ``sentence-transformers`` instalado (pip install sentence-transformers)
    - ``data/processed/games.csv`` existente

Em producao/demo com AWS, use ``LUDEX_EMBEDDING_PROVIDER=bedrock``.
Se nenhum provider estiver configurado, o script exibe uma mensagem
de erro e sai sem falhar.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"


def warmup_embeddings() -> None:
    print("🔎 Warmup de Embeddings Semanticos...")

    if not DATA_PATH.exists():
        print(f"❌ Arquivo nao encontrado: {DATA_PATH}")
        print("   Execute o pipeline de dados antes de gerar embeddings.")
        sys.exit(1)

    # Importar provider com fallback explicito
    try:
        from src.recommenders.embeddings import create_embedding_provider
    except ImportError as e:
        print(f"❌ Erro ao importar modulo de embeddings: {e}")
        sys.exit(1)

    provider = create_embedding_provider()
    if provider is None:
        print("⚠️  Nenhum provider de embeddings disponivel.")
        print("   Para local: LUDEX_EMBEDDING_PROVIDER=local LUDEX_ALLOW_MODEL_DOWNLOAD=1")
        print("   Para Bedrock: LUDEX_EMBEDDING_PROVIDER=bedrock com AWS configurado.")
        sys.exit(1)

    print(f"📦 Provider: {provider.provider_id} ({provider.dimension} dimensoes)")

    # Carregar dados
    games = pd.read_csv(DATA_PATH)
    print(f"📊 Catalogo: {len(games)} jogos")

    # Gerar embeddings
    from src.recommenders.semantic import SemanticRecommender

    start_time = time.time()
    recommender = SemanticRecommender(provider=provider)
    recommender.fit(games)
    duration = time.time() - start_time

    if recommender.is_active:
        print(f"✅ Embeddings gerados em {duration:.2f}s.")
        print(f"   Shape: {recommender.embeddings.shape}")
        print(f"   Cache persistido em: {recommender.cache_dir}")
    else:
        print("❌ Falha ao gerar embeddings.")
        sys.exit(1)


if __name__ == "__main__":
    warmup_embeddings()
