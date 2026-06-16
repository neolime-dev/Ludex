import pandas as pd
import os
from datasets import load_dataset

def add_steam_images():
    print("Iniciando mapeamento de imagens da Steam (Design Premium)...")
    
    # 1. Carregar o nosso Super Dataset atual (RAWG + IGDB)
    if not os.path.exists('data/processed/games.csv'):
        print("Erro: games.csv não encontrado.")
        return
    df_ludex = pd.read_csv('data/processed/games.csv')
    
    # 2. Carregar o dataset da Steam (FronkonGames) para pegar IDs e Imagens
    print("Baixando mapeamento de imagens do Hugging Face...")
    ds_steam = load_dataset('FronkonGames/steam-games-dataset', split='train')
    df_steam = pd.DataFrame({
        'title': ds_steam['name'],
        'steam_appid': ds_steam['appID'],
        'header_image': ds_steam['header_image']
    })
    
    # Limpeza básica no mapeamento
    df_steam = df_steam.drop_duplicates(subset=['title'])
    
    # 3. Merge por título (case-insensitive e strip)
    df_ludex['title_clean'] = df_ludex['title'].str.lower().str.strip()
    df_steam['title_clean'] = df_steam['title'].str.lower().str.strip()
    
    print("Fundindo imagens ao catálogo...")
    # Fazemos um left join para manter todos os jogos do Ludex
    df_final = pd.merge(df_ludex, df_steam[['title_clean', 'steam_appid', 'header_image']], on='title_clean', how='left')
    
    # 4. Fallback para quem não tem imagem: Imagem padrão de "No Image" ou placeholder
    df_final['header_image'] = df_final['header_image'].fillna("https://via.placeholder.com/460x215?text=Ludex+Gaming")
    
    # 5. Limpar e salvar
    df_final.drop(columns=['title_clean'], inplace=True)
    df_final.to_csv('data/processed/games.csv', index=False)
    print(f"Sucesso! Imagens integradas. Total de jogos: {len(df_final)}")

if __name__ == "__main__":
    add_steam_images()
