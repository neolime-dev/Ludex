import pandas as pd
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender

df = pd.read_csv('data/processed/games.csv')
cb = ContentBasedRecommender().fit(df)
hr = HybridRecommender(content_recommender=cb)

queries = ["roguelike desafiador", "jogo de fazenda relaxante", "cyberpunk neon"]

for q in queries:
    print(f"\n--- Testando busca: '{q}' ---")
    recs = hr.recommend(df, query=q, top_n=3)
    print(recs[['title', 'genres', 'score']])
