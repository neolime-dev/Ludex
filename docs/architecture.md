# Arquitetura do Ludex

Atualizado em 2026-06-21.

## Visao geral

O Ludex e composto por quatro camadas:

1. Ingestao e enriquecimento de dados em `src/data/`.
2. Contrato tabular em `data/processed/games.csv`.
3. Recomendadores em `src/recommenders/`.
4. Interface Streamlit em `app/streamlit_app.py`.

O MVP 1 usa TF-IDF como motor principal por ser barato, local, explicavel e rapido de iterar. A fase 2 atual prioriza uma demo offline-first: o ranking continua em TF-IDF/hibrido, e o Ollama atua como camada opcional de conversa/explicacao em cima dos jogos recuperados.

## Fluxo de dados

`src/data/download_and_prep.py` baixa e processa dados RAWG via Hugging Face.

`src/data/super_dataset_merge.py` tenta complementar descricoes com dados IGDB.

`src/data/add_images.py` adiciona `steam_appid` e `header_image`, filtra termos NSFW simples e prioriza jogos conhecidos para melhorar a primeira impressao da vitrine.

`src/data/load_data.py` carrega `data/processed/games.csv` e valida as colunas obrigatorias.

O app aplica uma segunda validacao em `validate_contract()`, normalizando tipos e preservando apenas colunas obrigatorias + opcionais conhecidas.

## Contrato do catalogo

Colunas obrigatorias:

- `game_id`: identificador estavel em string ou numerico.
- `title`: titulo exibido e usado no corpus.
- `genres`: lista textual separada por virgula, barra, pipe ou ponto e virgula.
- `tags`: mecanicas/temas do jogo.
- `description`: descricao textual.
- `release_year`: ano de lancamento.
- `positive_ratio`: nota percentual de avaliacao positiva, de 0 a 100.

Colunas opcionais usadas pelo app:

- `price`
- `developer`
- `publisher`
- `url_store`
- `url_ref`
- `header_image`
- `steam_appid`
- `review_keywords`
- `sentiment_score`

Se novas colunas entrarem na fase 2, prefira adiciona-las como opcionais e manter retrocompatibilidade com o CSV atual.

## Recomendador TF-IDF

`ContentBasedRecommender` treina duas matrizes:

- `matrix`: conteudo de titulo, generos, tags e descricao.
- `opinion_matrix`: conteudo semelhante, mas com mais peso para `review_keywords`.

Pesos atuais do corpus principal:

- titulo: 1x
- generos: 5x
- tags: 10x
- descricao: 1x

O objetivo e forcar mecanicas e temas a pesarem mais do que texto narrativo longo. O metodo `score_by_game_ids()` calcula um centroide vetorial quando o usuario seleciona varios jogos de referencia.

## Recomendador hibrido

`HybridRecommender` combina:

- `content_score`: similaridade com jogos de referencia.
- `opinion_score`: similaridade da busca textual contra corpus opinativo.
- `quality_score`: mistura de `positive_ratio` e `sentiment_score`, quando disponivel.

Pesos base:

- conteudo: 0.70
- opiniao/busca: 0.15
- qualidade: 0.15

Os pesos sao normalizados dinamicamente. Se nao houver busca nem referencia, qualidade vira fallback principal.

## Interface

`app/streamlit_app.py` renderiza:

- hero e status do catalogo;
- campo de busca por intencao;
- multiselect de jogos de referencia;
- filtros por genero, tag, ano, qualidade, preco e desenvolvedor;
- cards com imagem, explicacao amigavel, links externos e match visual;
- detalhes tecnicos de ranking recolhidos por padrao;
- aba `Assistente Ludex` com status claro de Ollama ativo ou fallback local.

A busca textual e importante para demonstracoes porque permite exemplos naturais como `roguelike desafiador` e `jogo de fazenda relaxante`. Nao remova esse campo sem substituir por UX equivalente.

## Cache e desempenho

Existem dois niveis de cache:

- `st.cache_data` e `st.cache_resource` dentro do Streamlit para a sessao em execucao.
- `data/processed/cache/tfidf_model.pkl`, gerado por `scripts/warmup_cache.py`.

O app carrega o pickle apenas se:

- o arquivo existe;
- o pickle esta mais novo que `data/processed/games.csv`;
- o objeto carregado e um `ContentBasedRecommender`;
- `model_cache_version` bate com `RECOMMENDER_MODEL_VERSION`;
- os `game_id` do cache batem com os `game_id` atuais na mesma ordem.

Se qualquer validacao falhar, o app treina o TF-IDF em memoria. Depois de alterar dados, pesos ou estrutura do recomendador, execute:

```bash
python scripts/warmup_cache.py
```

## Embeddings e provedores opcionais

O componente semantico fica isolado em `src/recommenders/semantic.py` e recebe um `EmbeddingProvider`. A factory em `src/recommenders/embeddings/__init__.py` e conservadora por desenho:

- `LUDEX_EMBEDDING_PROVIDER=auto` e o padrao. Nesse modo, tenta Bedrock apenas quando ha indicio de configuracao AWS. Bedrock e legado/opcional, nao requisito da demo local.
- O provider local (`sentence-transformers`) e opt-in com `LUDEX_EMBEDDING_PROVIDER=local`, porque um cache incompleto do Hugging Face pode disparar retries de rede e atrasar a demo.
- Downloads do modelo local so devem acontecer com `LUDEX_ALLOW_MODEL_DOWNLOAD=1`.
- Sem provider ativo, `HybridRecommender` recebe score semantico zero e normaliza os pesos sem quebrar o ranking TF-IDF.

Essa politica preserva o fallback offline: a interface deve abrir com TF-IDF mesmo sem AWS, sem internet e sem cache semantico valido.

## Fase 2 atual: IA Local e RAG offline-first

A arquitetura evoluiu para não depender de APIs pagas na nuvem (AWS/Bedrock).

1. O componente de IA Generativa foi internalizado com o módulo `src/recommenders/ollama_explainer.py`.
2. `OllamaExplainer` lê `LUDEX_OLLAMA_URL` e `LUDEX_OLLAMA_MODEL`, com defaults `http://localhost:11434` e `ludex-assistant:latest`.
3. O health check consulta `/api/tags` e só ativa o LLM quando o modelo esperado existe.
4. O ChatBot (Assistente Ludex) usa RAG: recebe a pergunta do usuário, junta com o histórico recente, consulta o `HybridRecommender` e envia somente o top recuperado como contexto ao Ollama.
5. Quando Ollama ou o modelo estão indisponíveis, o assistente responde por fallback determinístico com as recomendações do ranking local. A aba não bloqueia e não exige credenciais externas.
6. Foi implementada infraestrutura experimental de Fine-Tuning (Unsloth/LoRA) em `scripts/fine_tuning/` para treinar modelos Llama 3 localmente com o dataset do Ludex. Pesos, checkpoints, outputs e datasets gerados ficam ignorados no Git; apenas scripts, `Modelfile` e manifest pequeno devem ser versionados.

## Riscos atuais

- `review_keywords` pode ser sintetico ou incompleto, entao o score opinativo ainda e fraco para alguns jogos.
- Pickle nao e formato portavel entre grandes mudancas de classe; regenere cache quando mexer em `ContentBasedRecommender`.
- O dataset pode conter duplicatas semanticas por titulo/plataforma, mesmo quando `game_id` e unico.
- Embeddings via Bedrock introduzem custo, latencia e dependencia de credenciais. Trate como legado/opcional e use cache obrigatoriamente.
- Embeddings locais sao uteis para comparacao, mas nao devem ser inicializados implicitamente no app quando o modelo nao esta garantidamente completo no cache.
- Ollama pode estar desligado ou sem `ludex-assistant:latest`; a UI deve mostrar instrucao objetiva e manter fallback local.
