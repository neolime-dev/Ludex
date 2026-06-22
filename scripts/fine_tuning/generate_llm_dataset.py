import random
import json
from pathlib import Path

import pandas as pd

# Configurações de caminhos
DATA_PATH = Path("data/processed/games.csv")
OUTPUT_DIR = Path("data/processed/fine_tuning")
OUTPUT_FILE = OUTPUT_DIR / "ludex_instruction_dataset.jsonl"

def load_games():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def generate_synthetic_query(row):
    """Cria uma intenção de busca sintética baseada nas tags e gêneros do jogo."""
    genres = str(row.get('genres', '')).split(',')
    tags = str(row.get('tags', '')).split(',')
    
    # Limpa espaços
    genres = [g.strip() for g in genres if g.strip()]
    tags = [t.strip() for t in tags if t.strip()]
    
    templates = [
        "Estou procurando um jogo com os gêneros: {g}.",
        "Me recomende algo focado em {t}.",
        "Quero um jogo no estilo {g} que tenha elementos de {t}.",
        "Qual jogo você indica para quem gosta de {g}?",
        "Estou buscando uma experiência que envolva {t}.",
    ]
    
    g_choice = random.choice(genres) if genres else "jogos casuais"
    t_choice = random.choice(tags) if tags else "aventura"
    
    template = random.choice(templates)
    return template.format(g=g_choice, t=t_choice)

def generate_response(row):
    """Cria a resposta ideal e direta que o LLM deve dar."""
    title = row.get('title', 'Jogo Desconhecido')
    genres = clean_terms(row.get('genres', ''), limit=2)
    tags = clean_terms(row.get('tags', ''), limit=4)
    genres_text = ", ".join(genres) if genres else "um estilo forte do catálogo"
    tags_text = ", ".join(tags) if tags else "mecânicas compatíveis com a busca"

    templates = [
        f"Eu recomendaria '{title}'. Ele combina com a busca por estar em {genres_text} e trazer sinais como {tags_text}.",
        f"'{title}' é uma boa escolha nesse perfil: os gêneros {genres_text} e as tags {tags_text} apontam para a experiência pedida.",
        f"Para esse pedido, vale olhar '{title}'. A recomendação vem principalmente de {genres_text}, com destaque para {tags_text}.",
    ]
    return random.choice(templates)


def clean_terms(value, limit=4):
    terms = [
        term.strip()
        for term in str(value or '').replace('|', ',').replace(';', ',').split(',')
        if term.strip()
    ]
    return terms[:limit]

def build_dataset():
    print("Carregando games.csv...")
    df = load_games()
    
    # Filtra jogos inválidos e pega um subset relevante
    df = df.dropna(subset=['title', 'description'])
    if 'positive_ratio' in df.columns:
        df = df[df['positive_ratio'] > 70]
    
    # Embaralha o dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dataset_entries = []
    MAX_EXAMPLES = 10000 
    
    print(f"Gerando dataset com até {MAX_EXAMPLES} exemplos sintéticos...")
    for idx, row in df.head(MAX_EXAMPLES).iterrows():
        instruction = (
            "Você é o Assistente Ludex, um expert recomendador de jogos. "
            "Responda em português do Brasil, sem copiar sinopses ou descrições do catálogo. "
            "Use apenas título, gêneros e tags como base da justificativa."
        )
        input_text = generate_synthetic_query(row)
        output_text = generate_response(row)
        
        entry = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        dataset_entries.append(entry)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ Dataset de Fine-Tuning gerado com sucesso!")
    print(f"Salvo em: {OUTPUT_FILE}")
    print(f"Total de exemplos: {len(dataset_entries)}")

if __name__ == "__main__":
    build_dataset()
