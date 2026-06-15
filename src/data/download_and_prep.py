import ast
import gzip
import json
import os
import re

import pandas as pd
import requests


SENTIMENT_TO_POSITIVE_RATIO = {
    "Overwhelmingly Positive": 96,
    "Very Positive": 90,
    "Positive": 80,
    "Mostly Positive": 70,
    "Mixed": 50,
    "Mostly Negative": 35,
    "Negative": 25,
    "Very Negative": 15,
    "Overwhelmingly Negative": 5,
}


def parse_record(line):
    try:
        return ast.literal_eval(line)
    except (SyntaxError, ValueError):
        return json.loads(line)


def normalize_list(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    if pd.isna(value):
        return ""
    return str(value)


def parse_release_year(release_date):
    match = re.search(r"(19|20)\d{2}", str(release_date or ""))
    return int(match.group(0)) if match else 0


def normalize_price(price):
    if price is None:
        return 0.0
    text = str(price).strip()
    if not text or text.lower().startswith("free"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def sentiment_to_positive_ratio(sentiment):
    return SENTIMENT_TO_POSITIVE_RATIO.get(str(sentiment or "").strip(), 50)


def build_description(game):
    description = str(game.get("description") or game.get("about_the_game") or "").strip()
    if description:
        return description

    title = str(game.get("title") or game.get("app_name") or "").strip()
    genres = normalize_list(game.get("genres"))
    tags = normalize_list(game.get("tags"))
    specs = normalize_list(game.get("specs"))
    developer = str(game.get("developer") or "").strip()
    publisher = str(game.get("publisher") or "").strip()
    return " ".join(
        part
        for part in [
            title,
            f"Genres: {genres}" if genres else "",
            f"Tags: {tags}" if tags else "",
            f"Specs: {specs}" if specs else "",
            f"Developer: {developer}" if developer else "",
            f"Publisher: {publisher}" if publisher else "",
        ]
        if part
    )

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"Arquivo {dest} já existe. Pulando download.")
        return
    print(f"Baixando {url}...")
    response = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download concluído.")

import random

def generate_mock_reviews(positive_ratio):
    good_words = ["masterpiece", "amazing", "immersive", "fun", "great story", "addictive", "beautiful"]
    bad_words = ["clunky", "boring", "repetitive", "frustrating", "buggy", "hard", "terrible"]
    mixed_words = ["okay", "short", "niche", "grindy", "potential", "average"]
    
    ratio = float(positive_ratio)
    words = []
    
    # Semente baseada no ratio para manter consistência
    random.seed(int(ratio * 100))
    
    if ratio >= 80:
        words = random.sample(good_words, 2) + random.sample(mixed_words, 1)
    elif ratio <= 40:
        words = random.sample(bad_words, 2) + random.sample(mixed_words, 1)
    else:
        words = random.sample(mixed_words, 2) + random.sample(good_words, 1)
        
    random.seed() # reset seed
    return ", ".join(words)

def process_games(input_path, output_path):
    print("Processando jogos (proteção de memória ativa)...")
    games = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                game = parse_record(line)
                title = str(game.get('title') or game.get('app_name') or "").strip()
                game_id = str(game.get('id') or "").strip()
                if not title or not game_id:
                    continue
                
                p_ratio = sentiment_to_positive_ratio(game.get('sentiment'))

                games.append({
                    'game_id': game_id,
                    'title': title,
                    'genres': normalize_list(game.get('genres')),
                    'tags': normalize_list(game.get('tags')),
                    'description': build_description(game),
                    'release_year': parse_release_year(game.get('release_date')),
                    'positive_ratio': p_ratio,
                    'review_keywords': generate_mock_reviews(p_ratio),
                    'price': normalize_price(game.get('price')),
                    'developer': str(game.get('developer') or '').strip(),
                    'publisher': str(game.get('publisher') or '').strip(),
                    'url_store': f"https://store.steampowered.com/app/{game_id}",
                    'url_ref': f"https://www.pcgamingwiki.com/w/index.php?search={title.replace(' ', '+')}"
                })
            except Exception:
                continue
    
    df = pd.DataFrame(games)
    df = df.drop_duplicates(subset=['game_id'])
    df = df.sort_values(['positive_ratio', 'release_year', 'title'], ascending=[False, False, True])
    df.to_csv(output_path, index=False)
    print(f"Dataset processado salvo em {output_path}. Total: {len(df)} jogos.")

if __name__ == "__main__":
    RAW_PATH = "data/raw/steam_games.json.gz"
    PROCESSED_PATH = "data/processed/games.csv"
    URL = "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz"
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    download_file(URL, RAW_PATH)
    process_games(RAW_PATH, PROCESSED_PATH)
