import pandas as pd
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender

df = pd.read_csv('data/processed/games.csv')
mc_dungeons = df[df['title'].str.contains('Minecraft: Dungeons', case=False, na=False)]
if not mc_dungeons.empty:
    mc_id = mc_dungeons.iloc[0]['game_id']
    print(f"Encontrado: {mc_dungeons.iloc[0]['title']} (ID: {mc_id})")
    
    cb = ContentBasedRecommender().fit(df)
    hr = HybridRecommender(content_recommender=cb)
    
    recs = hr.recommend(df, reference_game_id=mc_id, top_n=5)
    print("\nRecomendações:")
    print(recs[['title', 'genres', 'score']])
    
    print("\nExplicação para a 1ª recomendação:")
    rec_id = recs.iloc[0]['game_id']
    terms = cb.top_terms_for_recommendation(rec_id, reference_game_id=mc_id)
    print(terms)
else:
    print("Minecraft: Dungeons não encontrado!")
