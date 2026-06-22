from pathlib import Path
import pickle
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommenders.content_based import ContentBasedRecommender

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "games.csv"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"
MODEL_CACHE = CACHE_DIR / "tfidf_model.pkl"


def warmup_and_cache():
    print("🚀 Iniciando warmup do recomendador TF-IDF...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 1. Carregar dados
    df = pd.read_csv(DATA_PATH)

    # 2. Treinar Recomendador
    print("🧠 Treinando matrizes TF-IDF...")
    recommender = ContentBasedRecommender().fit(df)

    # 3. Salvar Cache
    with MODEL_CACHE.open("wb") as f:
        pickle.dump(recommender, f)

    duration = time.time() - start_time
    print(f"✅ Warmup concluído em {duration:.2f}s.")
    print(f"📦 Cache persistido em: {MODEL_CACHE}")
    print("💡 O Streamlit carregara este cache quando ele for compativel com o catalogo.")

if __name__ == "__main__":
    warmup_and_cache()
