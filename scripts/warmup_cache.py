import pickle
import os
import time
import pandas as pd
from src.recommenders.content_based import ContentBasedRecommender

CACHE_DIR = "data/processed/cache"
MODEL_CACHE = os.path.join(CACHE_DIR, "tfidf_model.pkl")

def warmup_and_cache():
    print("🚀 Iniciando Warmup do Sistema (Gemini Performance Mode)...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # 1. Carregar dados
    df = pd.read_csv('data/processed/games.csv')
    
    # 2. Treinar Recomendador
    print("🧠 Treinando matrizes TF-IDF (N-Grams ativo)...")
    recommender = ContentBasedRecommender().fit(df)
    
    # 3. Salvar Cache
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(recommender, f)
    
    duration = time.time() - start_time
    print(f"✅ Warmup concluído em {duration:.2f}s.")
    print(f"📦 Cache persistido em: {MODEL_CACHE}")
    print("💡 O Streamlit agora carregará instantaneamente na sua aula!")

if __name__ == "__main__":
    warmup_and_cache()
