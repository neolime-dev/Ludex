import pandas as pd
import os

def create_super_dataset():
    print("Iniciando criação do Super Dataset (RAWG + IGDB)...")
    
    # 1. Carregar RAWG atual
    if not os.path.exists('data/processed/games.csv'):
        print("Erro: games.csv não encontrado. Execute download_and_prep.py primeiro.")
        return
    df_rawg = pd.read_csv('data/processed/games.csv')
    
    # 2. Baixar IGDB (PlayMyData - Versão PC)
    print("Baixando dados complementares do IGDB...")
    try:
        df_igdb = pd.read_csv('https://huggingface.co/datasets/claudioDsi94/PlayMyData/resolve/main/all_games_PC.csv')
    except Exception as e:
        print(f"Erro ao baixar IGDB: {e}")
        return

    # 3. Limpeza para o Merge
    df_igdb = df_igdb[['name', 'summary', 'storyline', 'rating']].rename(columns={
        'name': 'title',
        'summary': 'igdb_summary',
        'storyline': 'igdb_storyline',
        'rating': 'igdb_rating'
    })
    
    # 4. Merge (Fusão) - Usando 'title' como chave
    print("Fundindo datasets...")
    super_df = pd.merge(df_rawg, df_igdb, on='title', how='left')
    
    # 5. Enriquecimento: Se a descrição original for curta, usa o summary do IGDB
    def enrich_description(row):
        original = str(row['description'])
        igdb_text = str(row['igdb_summary']) if pd.notna(row['igdb_summary']) else ""
        if len(original) < 100 and len(igdb_text) > 100:
            return igdb_text
        return original

    super_df['description'] = super_df.apply(enrich_description, axis=1)
    
    # 6. Salvar
    super_df.drop(columns=['igdb_summary', 'igdb_storyline', 'igdb_rating'], inplace=True, errors='ignore')
    super_df = super_df.drop_duplicates(subset=['game_id'])
    super_df.to_csv('data/processed/games.csv', index=False)
    print(f"Super Dataset criado com sucesso! Total de jogos: {len(super_df)}")

if __name__ == "__main__":
    create_super_dataset()
