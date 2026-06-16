import pandas as pd
import os
from datasets import load_dataset

def add_steam_images():
    print("Iniciando mapeamento de imagens da Steam (Design Premium)...")
    
    if not os.path.exists('data/processed/games.csv'):
        print("Erro: games.csv não encontrado.")
        return
    df_ludex = pd.read_csv('data/processed/games.csv')
    
    # Remover colunas de imagem antigas se existirem para evitar duplicatas no merge (.x, .y)
    df_ludex = df_ludex.drop(columns=['header_image', 'steam_appid'], errors='ignore')
    
    print("Baixando mapeamento de imagens do Hugging Face...")
    ds_steam = load_dataset('FronkonGames/steam-games-dataset', split='train')
    df_steam = pd.DataFrame({
        'title': ds_steam['name'],
        'steam_appid': ds_steam['appID'],
        'header_image': ds_steam['header_image']
    })
    
    df_steam = df_steam.drop_duplicates(subset=['title'])
    
    df_ludex['title_clean'] = df_ludex['title'].str.lower().str.strip()
    df_steam['title_clean'] = df_steam['title'].str.lower().str.strip()
    
    print("Fundindo imagens ao catálogo...")
    df_final = pd.merge(df_ludex, df_steam[['title_clean', 'steam_appid', 'header_image']], on='title_clean', how='left')
    
    df_final['header_image'] = df_final['header_image'].fillna("https://via.placeholder.com/460x215?text=Ludex+Gaming")
    
    # --- FILTRO DE SEGURANÇA (NSFW) ---
    NSFW_KEYWORDS = ['hentai', 'adult', 'sexual', 'nudity', 'nsfw', 'porn', 'erotic', 'pregnation', 'sex']
    
    def is_safe(row):
        text = (str(row['title']) + " " + str(row.get('genres', '')) + " " + str(row.get('tags', ''))).lower()
        return not any(word in text for word in NSFW_KEYWORDS)

    print("Limpando catálogo (Filtro NSFW e Relevância)...")
    df_final = df_final[df_final.apply(is_safe, axis=1)]
    
    # --- PRIORIZAÇÃO DE JOGOS FAMOSOS (LANDING PAGE) ---
    FAMOUS_GAMES = [
        "Cyberpunk 2077", "Baldur's Gate 3", "Elden Ring", "The Witcher 3", 
        "Hades", "Portal 2", "Minecraft", "Terraria", "Stardew Valley", 
        "Counter-Strike", "Dota 2", "Grand Theft Auto V"
    ]
    
    df_final['is_famous'] = df_final['title'].apply(
        lambda x: 1 if any(fg.lower() in str(x).lower() for fg in FAMOUS_GAMES) else 0
    )
    
    # Ordenar: Famosos primeiro, depois por popularidade (positive_ratio)
    df_final = df_final.sort_values(by=['is_famous', 'positive_ratio'], ascending=[False, False])
    
    # --- BLINDAGEM FINAL (IDs ÚNICOS) ---
    # Removendo qualquer duplicata residual de game_id que possa quebrar o HybridRecommender
    df_final = df_final.drop_duplicates(subset=['game_id'], keep='first')
    
    # Limpeza final
    df_final.drop(columns=['title_clean', 'is_famous'], inplace=True)
    df_final.to_csv('data/processed/games.csv', index=False)
    print(f"Sucesso! Catálogo limpo e imagens integradas. Total: {len(df_final)} jogos.")

if __name__ == "__main__":
    add_steam_images()
