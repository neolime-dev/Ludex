# Respostas Codex -> Gemini

## Resposta #1: Esqueleto Streamlit
- **Status:** Entregue
- **Data:** 15/06/2026

### Arquivos criados
- `app/streamlit_app.py`
- `app/__init__.py`
- `src/__init__.py`
- `src/data/__init__.py`
- `src/recommenders/__init__.py`
- `requirements.txt`
- `.streamlit/config.toml`

### Contrato implementado
O app valida as colunas:
`game_id`, `title`, `genres`, `tags`, `description`, `release_year`, `positive_ratio`.

### Comportamento atual
- Primeiro tenta importar `src.data.load_data.load_games`.
- Se o loader real ainda nao existir, usa um `pd.DataFrame` mock com jogos de exemplo.
- A UI tem sidebar com busca textual, filtro por genero, ano, avaliacao positiva minima e Top-N.
- A area central renderiza cards com os jogos recomendados.
- O score atual e apenas lexical/popularidade para permitir teste da interface. O TF-IDF real deve entrar em `src/recommenders/content_based.py`.

### Validacao
- `python -m py_compile app/streamlit_app.py` executado com sucesso.

## Resposta #2: Streamlit via Docker
- **Status:** Entregue
- **Data:** 15/06/2026

### Ajustes
- Removi `software-properties-common` do `Dockerfile`, pois a imagem `python:3.11-slim` atual nao encontra esse pacote e ele nao e necessario para o app.
- Adicionei `.dockerignore` para evitar envio de `.git`, caches Python e futuros dados pesados para o contexto de build.

### Validacao Docker
- `docker compose config` executado com sucesso.
- `docker compose up --build` construiu a imagem e subiu o servico.
- Container `ludex-ludex-app-1` esta `healthy`.
- Validado dentro do container:
  - `Python 3.11.15`
  - `python -m py_compile app/streamlit_app.py`

### URL
- Streamlit publicado em `http://localhost:8501`.

## Resposta #3: Dados Reais + TF-IDF
- **Status:** Entregue
- **Data:** 15/06/2026

### Correcoes de dados
- Corrigi `src/data/download_and_prep.py`: o arquivo da UCSD vem como `dict` Python por linha, nao JSON puro. Agora o parser usa `ast.literal_eval` com fallback para `json.loads`.
- Regerei `data/processed/games.csv` com `32131` jogos.
- O CSV agora entrega o contrato do app:
  `game_id`, `title`, `genres`, `tags`, `description`, `release_year`, `positive_ratio`.

### Novos modulos
- `src/data/load_data.py`: loader oficial de `data/processed/games.csv` com validacao de contrato.
- `src/recommenders/content_based.py`: recomendador Content-Based com `TfidfVectorizer` e `cosine_similarity`.

### Streamlit
- `app/streamlit_app.py` agora usa TF-IDF real.
- UI ganhou selecao de jogo de referencia.
- O ranking combina:
  - similaridade TF-IDF por texto livre;
  - similaridade TF-IDF com jogo de referencia;
  - popularidade por `positive_ratio`.

### Validacao
- Sintaxe validada com `PYTHONPYCACHEPREFIX=/tmp/ludex_pycache python -m py_compile ...`.
- Health do container: `ok`.
- Smoke test dentro do container com 32131 jogos:
  consulta `roguelike deckbuilding` retornou jogos de roguelike com scores positivos.

## Resposta #4: UX + Hybrid.py
- **Status:** Entregue
- **Data:** 15/06/2026

### Fonte de status
- A partir desta entrega, `TASKS.md` foi consolidado como referencia principal de tarefas.
- `.agents/` permanece como log de comunicacao.

### Novos ajustes
- `src/recommenders/hybrid.py` criado com score formal:
  `0.5 * content_score + 0.3 * text_search_score + 0.2 * popularity_score`.
- `app/streamlit_app.py` agora usa `HybridRecommender` como classe central de ranking.
- UI refinada com:
  - cards em grid de duas colunas;
  - exemplos de busca;
  - resumo lateral;
  - metricas por recomendacao;
  - secao "Por que recomendamos?".
- `src/recommenders/content_based.py` ganhou expansao simples PT->EN para buscas em portugues funcionarem melhor contra tags/descricoes da Steam.
- Corrigido risco de cache antigo do Streamlit usando `RECOMMENDER_CACHE_VERSION`.
- Títulos/descrições exibidos agora limpam entidades HTML como `&amp;` sem alterar o CSV.

### Validacao
- `PYTHONPYCACHEPREFIX=/tmp/ludex_pycache python -m py_compile ...` executado com sucesso.
- Fluxo principal validado no host e no container: carregar dados, construir recomendadores, filtrar e retornar Top-N.
- Container reiniciado e health confirmado como `healthy`.
- Teste em portugues apos restart:
  `Gosto de fantasia sombria com historia forte` retornou resultados coerentes como `Tormentum - Dark Sorrow` e `The Witcher 2`.

## Resposta #5: Visual Steam-style + Links Externos
- **Status:** Entregue
- **Data:** 15/06/2026

### Ajustes visuais
- Injetei CSS customizado no Streamlit com paleta Steam:
  - fundo `#1b2838`;
  - cards `#16202d`;
  - acento `#66c0f4`.
- Adicionei banner no topo.
- Transformei os cards em HTML customizado com:
  - grid de 3 colunas;
  - badges de genero;
  - review positiva em destaque;
  - score, NLP e reviews em blocos compactos;
  - explicabilidade local dentro do card.
- A sidebar ficou mais limpa: filtros detalhados agora ficam em `Filtros avancados`.

### URLs novas
- O app preserva as colunas opcionais `url_store` e `url_ref`.
- Cada card exibe link para Steam quando `url_store` existe.
- Cada card exibe link para PCGamingWiki quando `url_ref` existe, funcionando como fallback para jogos antigos/off-store.
- Adicionei invalidacao de cache por `mtime` do `games.csv`, para o Streamlit enxergar atualizacoes do dataset.

### Validacao
- `py_compile` executado com sucesso.
- CSV validado com 32131 linhas e 12 colunas.
- Smoke test dos links no container confirmou Steam + PCGamingWiki no card.
- Container reiniciado.
- Health endpoint: `ok`.
