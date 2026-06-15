import os
import re
import pandas as pd
import random
from datasets import load_dataset

def generate_mock_reviews(positive_ratio):
    good_words = ["masterpiece", "amazing", "immersive", "fun", "great story", "addictive", "beautiful", "goty"]
    bad_words = ["clunky", "boring", "repetitive", "frustrating", "buggy", "hard", "terrible", "unoptimized"]
    mixed_words = ["okay", "short", "niche", "grindy", "potential", "average", "casual"]
    
    ratio = float(positive_ratio)
    
    random.seed(int(ratio * 100))
    if ratio >= 80:
        words = random.sample(good_words, 2) + random.sample(mixed_words, 1)
    elif ratio <= 40:
        words = random.sample(bad_words, 2) + random.sample(mixed_words, 1)
    else:
        words = random.sample(mixed_words, 2) + random.sample(good_words, 1)
    random.seed()
    
    return ", ".join(words)

def clean_html(raw_html):
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', str(raw_html)).strip()

def process_rawg_games(output_path):
    print("Baixando dataset RAWG do Hugging Face (IVproger/rawg-games-dataset-updated)...")
    dataset = load_dataset('IVproger/rawg-games-dataset-updated', split='train')
    
    print(f"Dataset carregado com {len(dataset)} jogos. Processando e limpando...")
    
    games = []
    # Usar um limite para não estourar a RAM, priorizando jogos com avaliações ou data de lançamento recente
    for i, row in enumerate(dataset):
        try:
            title = str(row.get('name', '')).strip()
            if not title:
                continue
                
            # RAWG usa rating de 0 a 5. Convertendo para percentual 0 a 100
            rating_raw = float(row.get('rating', 0) or 0)
            if rating_raw == 0 and i > 100000: # Manter apenas jogos sem nota se forem muito famosos (início da lista)
                continue
                
            p_ratio = (rating_raw / 5.0) * 100
            
            release_date = str(row.get('released', ''))
            match = re.search(r"(19|20)\d{2}", release_date)
            release_year = int(match.group(0)) if match else 0
            
            desc = clean_html(row.get('description', ''))
            genres = str(row.get('genres', '') or '')
            tags = str(row.get('tags', '') or '')
            
            # Filtro de qualidade: O jogo precisa ter descrição ou tags para o NLP funcionar bem
            if len(desc) < 20 and not tags:
                continue

            # Usamos um hash do título como game_id já que o RAWG não expõe o ID nativo de forma simples no CSV
            game_id = str(hash(title.lower()))
            
            games.append({
                'game_id': game_id,
                'title': title,
                'genres': genres,
                'tags': tags,
                'description': desc,
                'release_year': release_year,
                'positive_ratio': round(p_ratio, 2),
                'review_keywords': generate_mock_reviews(p_ratio),
                'price': 0.0, # RAWG dataset não tem preço, mantendo 0 por consistência do schema
                'developer': str(row.get('developers', '') or ''),
                'publisher': str(row.get('publishers', '') or ''),
                'url_store': f"https://rawg.io/games/{title.lower().replace(' ', '-')}",
                'url_ref': f"https://www.pcgamingwiki.com/w/index.php?search={title.replace(' ', '+')}"
            })
        except Exception:
            continue
            
    df = pd.DataFrame(games)
    df = df.drop_duplicates(subset=['title']) # RAWG pode ter duplicatas de plataformas
    
    # Ordenar por qualidade e atualidade
    df = df.sort_values(['positive_ratio', 'release_year'], ascending=[False, False])
    
    # Manter o Top 40k para performance do Streamlit e TF-IDF
    df = df.head(40000)
    
    df.to_csv(output_path, index=False)
    print(f"Sucesso! Dataset RAWG salvo em {output_path}. Total final: {len(df)} jogos de altíssima qualidade.")

if __name__ == "__main__":
    PROCESSED_PATH = "data/processed/games.csv"
    os.makedirs("data/processed", exist_ok=True)
    process_rawg_games(PROCESSED_PATH)
