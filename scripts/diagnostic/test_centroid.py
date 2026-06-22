from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_games
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender, HybridWeights
from scripts.diagnostic.benchmark_recommendations import try_build_semantic_recommender, format_results

def main():
    games = load_games()
    
    refs_list = [
        {"name": "Diablo II + Portal Knights", "ids": ["-7376267108040845681", "-6087521020158961950"]},
        {"name": "Path of Exile + Terraria", "ids": ["-722237789075736159", "-8997558743492711944"]},
        {"name": "Diablo + Dragon Quest Builders 2", "ids": ["678541679837678274", "8099004191437522922"]}
    ]
    
    content = ContentBasedRecommender().fit(games)
    hybrid_tfidf = HybridRecommender(content_recommender=content)
    
    semantic = try_build_semantic_recommender(games)
    if semantic:
        hybrid_semantic = HybridRecommender(
            content_recommender=content,
            semantic_recommender=semantic,
            weights=HybridWeights(content=0.35, semantic=0.35, opinion=0.15, quality=0.15)
        )
    else:
        print("Motor semântico não disponível")
        return

    for refs in refs_list:
        print(f"\n{'='*60}")
        print(f"Buscando centroide para: {refs['name']}")
        ref_ids = refs["ids"]
            
        print("\n--- TF-IDF Only ---")
        res_tfidf = hybrid_tfidf.recommend(games, reference_game_ids=ref_ids, top_n=10)
        res_tfidf_100 = hybrid_tfidf.recommend(games, reference_game_ids=ref_ids, top_n=100)
        res_tfidf_100.reset_index(drop=True, inplace=True)
        rank = res_tfidf_100[res_tfidf_100['title'].str.contains('Minecraft Dungeons', case=False, na=False)]
        
        print(format_results(res_tfidf))
        if not rank.empty:
            print(f"> Minecraft Dungeons está na posição #{rank.index[0] + 1} no ranking completo.")
        else:
            print("> Minecraft Dungeons não está no top 100.")
        
        print("\n--- Semântico + TF-IDF ---")
        res_sem = hybrid_semantic.recommend(games, reference_game_ids=ref_ids, top_n=10)
        res_sem_100 = hybrid_semantic.recommend(games, reference_game_ids=ref_ids, top_n=100)
        res_sem_100.reset_index(drop=True, inplace=True)
        rank_sem = res_sem_100[res_sem_100['title'].str.contains('Minecraft Dungeons', case=False, na=False)]
        
        print(format_results(res_sem))
        if not rank_sem.empty:
            print(f"> Minecraft Dungeons está na posição #{rank_sem.index[0] + 1} no ranking completo.")
        else:
            print("> Minecraft Dungeons não está no top 100.")

if __name__ == "__main__":
    main()
