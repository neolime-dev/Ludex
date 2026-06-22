# Ludex

Sistema de recomendacao de jogos com Streamlit, TF-IDF e ranking hibrido. O projeto combina dados de catalogo, tags, descricoes, sinais de reviews/comunidade e filtros interativos para sugerir jogos a partir de uma busca textual ou de jogos de referencia.

## Estado atual

- App principal em `app/streamlit_app.py`.
- Catalogo esperado em `data/processed/games.csv`.
- Motor atual: `ContentBasedRecommender` + `HybridRecommender`.
- Busca por intencao ativa na interface.
- Cache persistente TF-IDF opcional em `data/processed/cache/tfidf_model.pkl`.
- Fase 2 atual: Assistente Ludex offline-first com RAG local via Ollama opcional. O ranking segue obrigatório em TF-IDF/híbrido; detalhes técnicos ficam recolhidos na UI.

## Como executar

Local:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Docker:

```bash
docker compose up --build
```

Depois acesse:

```text
http://localhost:8501
```

## IA Local, RAG Autônomo e Fine-Tuning

O app abre sem AWS, sem internet e sem Ollama. O Assistente Ludex usa o ranking local do `HybridRecommender` como RAG; quando o Ollama e o modelo `ludex-assistant:latest` estão disponíveis, ele gera uma resposta conversacional em cima dos jogos recuperados. Quando não estão, responde em modo local determinístico com as melhores recomendações do catálogo.

Configuração padrão:

```bash
export LUDEX_OLLAMA_URL=http://localhost:11434
export LUDEX_OLLAMA_MODEL=ludex-assistant:latest
```

O ChatBot não substitui o ranking. Ele roda o `HybridRecommender` em tempo real baseado no histórico recente da conversa e usa esses resultados como contexto.

### Treinamento do Modelo (Fine-Tuning)
Você pode gerar o seu próprio modelo especializado usando a sua GPU (ex: RTX 4070) rodando os scripts que preparamos:
```bash
python scripts/fine_tuning/generate_llm_dataset.py
python scripts/fine_tuning/train_unsloth.py
ollama create ludex-assistant -f ludex-llama3-finetuned_gguf/Modelfile
```

## Contrato minimo dos dados

`data/processed/games.csv` precisa conter:

- `game_id`
- `title`
- `genres`
- `tags`
- `description`
- `release_year`
- `positive_ratio`

Campos opcionais usados pela UI:

- `price`
- `developer`
- `publisher`
- `url_store`
- `url_ref`
- `header_image`
- `steam_appid`
- `review_keywords`
- `sentiment_score`

## Verificacoes rapidas

O projeto ainda nao tem `pytest` configurado. Use os scripts abaixo como smoke tests:

```bash
python test_recommender.py
python test_text_search.py
python scripts/diagnostic/benchmark_recommendations.py
python scripts/diagnostic/test_recommender.py
python scripts/diagnostic/test_tfidf.py
python scripts/warmup_cache.py
```

## Fase 2

A prioridade da fase 2 é uma demo local robusta, com benchmark de qualidade antes de mudanças grandes no motor. Cenários fixos:

- `roguelike desafiador`
- `jogo de fazenda relaxante`
- `cyberpunk neon`
- `rpg narrativo com escolhas`
- referencias: `Minecraft Dungeons`, `Hades`, `Stardew Valley`, `Disco Elysium`

Meta da fase 2 atual:

- manter TF-IDF/híbrido como ranking obrigatório;
- usar Ollama apenas como camada opcional de conversa e explicação;
- manter fallback offline determinístico no Assistente Ludex;
- ignorar pesos/checkpoints/datasets de fine-tuning no Git;
- comparar qualquer motor semântico futuro contra os benchmarks em `docs/benchmark_latest.txt`.

Bedrock e outros provedores de embeddings permanecem legado/opcionais para experimentos, não requisito da demo local.

Leia `AGENTS.md`, `TASKS.md` e `docs/architecture.md` antes de continuar a implementacao.
