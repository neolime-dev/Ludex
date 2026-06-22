# Ludex - Guia para Agentes de IA

Atualizado em 2026-06-21.

## Contexto rapido

Ludex e um app Streamlit de recomendacao de jogos. O MVP atual usa um recomendador hibrido baseado em TF-IDF, sinais de reviews/comunidade e filtros de catalogo. A fase 2 deve melhorar qualidade percebida e apresentacao com embeddings semanticos via Amazon Bedrock, justificativas por LLM e cache semantico persistente.

O usuario relatou decepcao com o desempenho na apresentacao. Trate isso como prioridade de produto: antes de adicionar recursos grandes, crie ou rode cenarios de avaliacao que mostrem se as recomendacoes melhoraram.

## Estado atual do codigo

- Entrada principal: `app/streamlit_app.py`.
- Dados esperados: `data/processed/games.csv`.
- Contrato minimo do CSV: `game_id`, `title`, `genres`, `tags`, `description`, `release_year`, `positive_ratio`.
- Campos opcionais usados pela UI: `price`, `developer`, `publisher`, `url_store`, `url_ref`, `header_image`, `steam_appid`, `review_keywords`, `sentiment_score`.
- Recomendador de conteudo: `src/recommenders/content_based.py`.
- Recomendador hibrido: `src/recommenders/hybrid.py`.
- Cache persistente TF-IDF: `scripts/warmup_cache.py` gera `data/processed/cache/tfidf_model.pkl`.
- O app tenta carregar esse pickle quando ele esta mais novo que `games.csv`, tem a mesma versao de modelo e contem os mesmos `game_id` na mesma ordem; caso contrario, treina o TF-IDF em memoria.
- A busca textual esta ativa no app e alimenta o score opinativo/textual do recomendador hibrido.

## Como rodar

```bash
streamlit run app/streamlit_app.py
```

Ou com Docker:

```bash
docker compose up --build
```

Para aquecer o cache local antes de apresentacao:

```bash
python scripts/warmup_cache.py
```

## Verificacoes recomendadas

O projeto ainda nao tem suite `pytest` configurada. Use estes comandos como smoke tests:

```bash
python test_recommender.py
python test_text_search.py
python scripts/diagnostic/benchmark_recommendations.py
python scripts/diagnostic/test_recommender.py
python scripts/diagnostic/test_tfidf.py
python scripts/warmup_cache.py
```

Se adicionar testes formais, inclua `pytest` em `requirements.txt` ou crie uma secao clara de dependencias de desenvolvimento.

## Prioridade da Fase 2

1. Criar baseline de qualidade antes de trocar o motor:
   - consultas textuais: `roguelike desafiador`, `jogo de fazenda relaxante`, `cyberpunk neon`, `rpg narrativo com escolhas`;
   - perfis por referencia: `Minecraft Dungeons`, `Hades`, `Stardew Valley`, `Disco Elysium`;
   - registrar top 5, latencia e explicacao gerada.
2. Implementar embeddings semanticos de forma paralela ao TF-IDF, sem remover o MVP atual.
3. Persistir embeddings e metadados de cache em `data/processed/cache/`.
4. Adicionar reranking hibrido: semantico + TF-IDF/tags + qualidade/comunidade.
5. Gerar justificativas com LLM somente depois que o ranking estiver aceitavel.

## Coordenacao entre IAs

Existe uma pasta `.agents/` com logs transitorios de troca entre agentes. Ela nao deve ter outro `AGENTS.md`; o arquivo oficial e este, na raiz do projeto. Use `.agents/README.md` para explicar o quadro temporario, `.agents/*_to_*.md` para mensagens em andamento e `AGENTS.md`, `TASKS.md` e `docs/architecture.md` para instrucoes consolidadas.

O usuario tambem tem Antigravity aberto no tmux, em panes `agy`, com acesso a Claude Opus/Sonnet 4.6 e Gemini 3.1 Pro. Use essa capacidade como apoio de equipe quando fizer sentido, sem duplicar contexto:

- Codex deve continuar como integrador no repo: editar arquivos, manter escopo, rodar smoke tests e validar diffs.
- Antigravity/Claude/Gemini devem ser usados para analises paralelas de alto valor: revisao de arquitetura, diagnostico de ranking, propostas de prompts/Bedrock, analise de dados e revisao critica.
- Antes de pedir ajuda a outro agente, envie uma tarefa pequena e autocontida: objetivo, arquivos relevantes, comando de validacao e formato esperado da resposta.
- Prefira pedir saidas em forma de achados, patch sugerido ou plano curto; nao peça reescritas grandes sem benchmark.
- Ao receber proposta de outro agente, valide localmente antes de aplicar. Nao copie mudancas grandes sem entender impacto no contrato de dados e nos benchmarks.
- Evite gastar chamadas fortes em trabalho mecanico. Use modelos mais capazes para decisao tecnica e modelos mais baratos para resumo, classificacao e revisao textual.
- Registre decisoes aceitas em `TASKS.md` ou `docs/architecture.md`; registre conversas temporarias em `.agents/`.

Divisao recomendada para a fase 2:

- Claude Opus/Sonnet: revisar desenho do reranking semantico, prompts de justificativa e tradeoffs de UX/explicabilidade.
- Gemini Pro: analisar dataset, qualidade de tags/reviews, benchmark e estrategias de cache/embeddings.
- Codex: implementar o modulo, integrar no Streamlit, preservar fallback offline e rodar verificacoes.

## Cuidados ao editar

- Nao coloque chaves AWS, tokens ou segredos no repositorio. Use `.env` local e variaveis de ambiente.
- Evite alterar PDFs, `.aux`, `.log`, `.toc` e documentos academicos gerados, salvo pedido explicito.
- Preserve arquivos nao versionados do usuario em `docs/` e testes soltos.
- `data/raw/*` e `data/processed/*` estao no `.gitignore`, mas `data/processed/games.csv` ja aparece no historico. Nao remova nem recrie dados grandes sem necessidade.
- Se mudar o contrato do CSV, atualize `README.md`, `docs/architecture.md`, `TASKS.md` e este arquivo.
- Se mudar pesos do ranking, registre a motivacao e rode os cenarios de baseline.

## Pontos conhecidos

- O cache pickle e local e dependente da classe Python atual; se a estrutura de `ContentBasedRecommender` ou `RECOMMENDER_MODEL_VERSION` mudar, rode `python scripts/warmup_cache.py` de novo.
- `scripts/warmup_cache.py` nao baixa dados; ele assume que `data/processed/games.csv` existe.
- O projeto usa `boto3` em `requirements.txt`, mas ainda nao ha modulo Bedrock implementado.
- A qualidade dos dados de reviews ainda e limitada: `review_keywords` pode ser sintetico ou incompleto dependendo da origem do catalogo.
