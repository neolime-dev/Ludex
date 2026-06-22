# Relatório Final — Projeto Ludex

---

## Cabeçalho

| Campo | Valor |
|---|---|
| **Tipo do Documento** | Relatório Final |
| **Nome do Projeto** | Ludex — Sistema de Recomendação de Jogos Multiestratégia |
| **Gerente do Projeto** | Matheus Lima Ribeiro |

---

## Objetivos deste documento

Este documento fornece uma visão geral do desempenho do projeto **Ludex**, um motor de recomendação de jogos baseado em Processamento de Linguagem Natural (NLP), abrangendo a avaliação de qualidade dos dados, as entregas realizadas, os problemas enfrentados e as ações corretivas adotadas, em conformidade com o framework de Análise e Qualidade de Dados (Wang & Strong, 1996) e a metodologia TQM (Deming, 1982).

---

## Identificação do Projeto

O **Ludex** (código interno: `neolime-dev/Ludex`) é um sistema de recomendação multiestratégia para jogos eletrônicos, desenvolvido como projeto aplicado da disciplina de **Análise e Qualidade de Dados**. O sistema utiliza técnicas avançadas de NLP — especificamente **TF-IDF com N-Grams** (bigramas) e modelo **Híbrido ponderado** — para recomendar títulos com base em mecânicas de jogabilidade, gêneros e sentimento da comunidade, superando a limitação de recomendações superficiais baseadas apenas em títulos ou estética.

**Características técnicas resumidas:**

| Atributo | Valor |
|---|---|
| Linguagem Principal | Python 3.11 |
| Framework de Interface | Streamlit ≥ 1.35 |
| Motor NLP | TF-IDF (scikit-learn ≥ 1.3) com N-Grams (1, 2) |
| Infraestrutura | Docker (containerização completa) |
| Volume do Dataset Final | **38.918 registros** × **15 atributos** |
| Fontes de Dados | RAWG (tags de comunidade) + IGDB (descrições ricas) + Steam (imagens e IDs) |
| Arquivo de Dados | `data/processed/games.csv` (~46,4 MB) |
| Repositório | `https://github.com/neolime-dev/Ludex` |

---

## Desempenho do Projeto

O projeto Ludex aplicou integralmente a metodologia **TQM (Total Quality Management)**, conforme o ciclo de melhoria contínua de Deming, nas seguintes fases:

1. **Importação e Conhecimento Inicial:** Ingestão de +120.000 registros brutos a partir do dataset RAWG (via Hugging Face), seguida de visualização preliminar da estrutura e esquema de dados.

2. **Data Profiling (Perfilamento):** Análise automatizada de campos nulos, duplicidades, inconsistências de formato (HTML em descrições, datas inválidas) e outliers em ratings, realizada via pipeline Python e Jupyter Notebook ([01_model_evaluation.ipynb](file:///home/lemondesk/Dev/projects/Ludex/notebooks/01_model_evaluation.ipynb)).

3. **Medição de Indicadores (Métricas de Qualidade):** Aplicação das três métricas fundamentais conforme o framework da disciplina:

   | Indicador | Fórmula | Valor Pré-Saneamento | Valor Pós-Saneamento |
   |---|---|---|---|
   | **Completude** | (Campos Preenchidos / Campos Totais) × 100 | ~72% (120k registros brutos com alta nulidade) | **89,65%** (dataset final processado) |
   | **Unicidade (title)** | (Registros Únicos / Registros Totais) × 100 | ~85% (duplicatas de plataforma no RAWG) | **100,00%** |
   | **Unicidade (game_id)** | (IDs Únicos / Total de IDs) × 100 | ~92% (colisões na fusão RAWG+IGDB) | **100,00%** |
   | **Validade (positive_ratio)** | (Dados Válidos 0-100 / Total) × 100 | — | **100,00%** |
   | **Validade (release_year)** | (Anos válidos 1970-2027 / Total) × 100 | — | **99,30%** |

4. **Diagnóstico de Impactos:** Identificação de que 16,82% dos gêneros e 55,35% dos publishers estavam nulos nas fontes originais, comprometendo a capacidade de filtragem e curadoria por editora.

5. **Limpeza e Saneamento (Data Cleansing):** Execução de pipelines automatizados de correção:
   - Remoção de HTML das descrições (`clean_html`)
   - Eliminação de duplicatas por `title` e `game_id` (`drop_duplicates`)
   - Filtro de conteúdo NSFW (9 keywords de segurança)
   - Enriquecimento cruzado de descrições curtas via IGDB (`enrich_description`)
   - Normalização de ratings RAWG (escala 0-5 → percentual 0-100)
   - Filtro de relevância: exclusão de jogos sem descrição (<20 chars) e sem tags

6. **Reavaliação:** Recálculo completo dos indicadores no dataset final (`data/processed/games.csv`), confirmando a evolução métrica documentada na tabela acima.

A aplicação do ciclo PDCA (Plan-Do-Check-Act) de Deming garantiu que o saneamento não fosse um evento pontual, mas um processo iterativo validado quantitativamente em cada fase.

---

## Desempenho em relação às entregas previstas

| Entrega | Critérios de aceitação verificados |
|---|---|
| **Pipeline de Ingestão de Dados** (`download_and_prep.py`) | ✅ Ingestão automatizada de +120k registros do RAWG via Hugging Face. Limpeza de HTML, normalização de ratings e filtro de relevância por qualidade. |
| **Fusão de Datasets (Super Dataset)** (`super_dataset_merge.py`) | ✅ Merge LEFT JOIN entre RAWG e IGDB por título. Enriquecimento condicional de descrições curtas (<100 chars) com summaries do IGDB. |
| **Integração de Imagens** (`add_images.py`) | ✅ Mapeamento de imagens da Steam via Hugging Face. Filtro NSFW aplicado. Placeholder para jogos sem capa. Blindagem final de IDs únicos. |
| **Motor de Recomendação Content-Based** (`content_based.py`) | ✅ TF-IDF com 30.000 features, N-Grams (1,2), pesos calibrados (Tags 10x, Gêneros 5x). Suporte a busca textual, similaridade por jogo e perfil multi-jogo (centroide vetorial). |
| **Motor Híbrido** (`hybrid.py`) | ✅ Ponderação adaptativa: Content (70%), Opinion (15%), Quality (15%). Fallback multinível para Out-of-Vocabulary. Normalização de sentimento e score de qualidade composto. |
| **Explainable AI (XAI)** (`top_terms_for_recommendation`) | ✅ Decomposição do cosseno em contribuições por termo TF-IDF. Exibição dos "porquês" de cada recomendação. |
| **Interface Streamlit Premium** (`streamlit_app.py`) | ✅ Design "Steam-Style" com CSS customizado, modo escuro, ícones Lucide, cards de alta fidelidade. 54 KB de código de interface. |
| **Cache Persistente** (`warmup_cache.py`) | ✅ Serialização do modelo treinado em `.pkl`. Redução do tempo de carregamento de ~30s para <1s. |
| **Infraestrutura Docker** (`Dockerfile` + `docker-compose.yml`) | ✅ Container Python 3.11-slim com healthcheck. Reprodutibilidade garantida em qualquer ambiente. |
| **Validação Acadêmica** (`01_model_evaluation.ipynb`) | ✅ Notebook de análise exploratória e avaliação do modelo para fundamentação acadêmica. |

---

## Desempenho em relação ao prazo e ao orçamento previsto

O MVP 1 do Ludex foi entregue dentro do prazo estipulado (16/06/2026), com todas as entregas concluídas e validadas conforme registrado no quadro de tarefas do projeto ([TASKS.md](file:///home/lemondesk/Dev/projects/Ludex/TASKS.md)). O projeto não incorreu em custos financeiros diretos, utilizando exclusivamente APIs gratuitas (Hugging Face Datasets), frameworks open-source (Streamlit, scikit-learn) e infraestrutura local (Docker). A Fase 2 (integração com Amazon Bedrock) está planejada para 22/06/2026.

---

## Principais problemas enfrentados

| Problema | Resolução adotada e recomendações futuras |
|---|---|
| **Alta nulidade em campos-chave** (Gêneros: 16,82%, Publishers: 55,35%, Tags: 3,84%) | Implementação de enriquecimento cruzado via fusão de datasets (RAWG + IGDB). O campo `publisher` permanece com alta nulidade por não ser crítico para o motor NLP; recomenda-se integração com a API da Steam para saneamento futuro. |
| **Duplicatas inter-dataset** (plataforma e colisões de ID na fusão) | Aplicação de `drop_duplicates` em múltiplas camadas: por `title` no RAWG, por `game_id` no merge final, e por `title_clean` no mapeamento de imagens. Unicidade final: **100%**. |
| **Conteúdo NSFW nos dados de origem** | Implementação de filtro de segurança com 9 keywords aplicado sobre título, gêneros e tags combinados. Varredura preventiva em toda a base antes da publicação. |
| **Descrições insuficientes para NLP** (~2,57% com <20 caracteres) | Filtro de qualidade no pipeline de ingestão: jogos sem descrição mínima E sem tags são excluídos. Descrições curtas (<100 chars) são substituídas por summaries do IGDB quando disponíveis. |
| **Frieza Semântica do TF-IDF** (Out-of-Vocabulary para termos em português) | Implementação de dicionário de **Query Expansion** (PT→EN) com 40+ mapeamentos semânticos (ex: "ação" → "action fast-paced combat"). Garante que buscas em português retornem resultados relevantes. |
| **Latência de inicialização** (~30s para treinar TF-IDF em 38.918 registros) | Sistema de cache persistente com serialização Pickle. O modelo treinado é salvo em `data/processed/cache/tfidf_model.pkl` e carregado instantaneamente nas sessões subsequentes. |

---

## Questões em Aberto

1. **Completude do campo `publisher`:** Com 55,35% de nulidade, este campo não é utilizado no motor de recomendação atual, mas seria necessário para futuras funcionalidades de filtragem por editora. Recomenda-se integração com a API SteamSpy ou SteamWorks.

2. **Evolução para Embeddings Densos:** A Fase 2 prevê a substituição do TF-IDF por Amazon Titan Embeddings (via Bedrock), o que eliminará a limitação de Out-of-Vocabulary e permitirá similaridade semântica profunda.

3. **Validação com usuários reais:** O projeto ainda não passou por testes de aceitação com uma amostra de jogadores. Recomenda-se a aplicação de métricas de satisfação (ex: NDCG@10, Hit Rate) em uma versão beta pública.

4. **Governança contínua:** Conforme o framework TQM, o saneamento de dados não é um evento único. Recomenda-se a criação de um pipeline automatizado de re-ingestão e revalidação periódica (Data Stewardship) para manter a integridade dos ativos informacionais.

---

## Informações adicionais

### Mapeamento do Projeto nas Dimensões de Qualidade (Wang & Strong, 1996)

| Categoria | Dimensão | Aplicação no Ludex |
|---|---|---|
| **Intrínseca** | Precisão, Credibilidade | Ratings normalizados de 0-100 com validação de range. Fontes de dados consolidadas (RAWG + IGDB) com filtragem de outliers. |
| **Contextual** | Relevância, Completude, Atualização | Filtro de relevância (top 40k por rating). Completude de 89,65%. Dados atualizados até 2026. |
| **Representacional** | Interpretabilidade, Consistência | Normalização de formatos (HTML → texto plano, ratings 0-5 → 0-100). Schema validado em `load_data.py` com 7 colunas obrigatórias. |
| **Acessibilidade** | Acesso e Segurança | Filtro NSFW implementado. Interface Streamlit com acesso via porta 8501. Containerização Docker para reprodutibilidade. |
| **Big Data (3 Vs)** | Volume, Variedade | +120k registros ingeridos, refinados para 38.918. Três fontes heterogêneas (RAWG, IGDB, Steam) fundidas em schema único. |

### Stack Tecnológica Completa

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| NLP/ML | scikit-learn (TF-IDF, Cosine Similarity), NumPy |
| Dados | Pandas, Hugging Face Datasets |
| Interface | Streamlit (Custom CSS, Lucide Icons) |
| Infraestrutura | Docker, Docker Compose |
| Versionamento | Git, GitHub |
| Cache | Pickle (serialização de modelos) |

### Arquitetura de Módulos

```
Ludex/
├── src/
│   ├── data/
│   │   ├── download_and_prep.py    # Ingestão e limpeza RAWG
│   │   ├── super_dataset_merge.py  # Fusão RAWG + IGDB
│   │   ├── add_images.py           # Imagens Steam + Filtro NSFW
│   │   └── load_data.py            # Validação de schema
│   └── recommenders/
│       ├── content_based.py        # Motor TF-IDF (30k features)
│       └── hybrid.py               # Ponderação adaptativa
├── app/
│   └── streamlit_app.py            # Interface Premium (54 KB)
├── scripts/
│   ├── warmup_cache.py             # Cache persistente
│   └── diagnostic/                 # Testes de validação
├── data/
│   ├── raw/                        # Dados brutos (Steam JSON)
│   └── processed/
│       ├── games.csv               # Dataset final (38.918 jogos)
│       └── cache/                  # Modelo TF-IDF serializado
├── notebooks/
│   └── 01_model_evaluation.ipynb   # Análise exploratória
├── Dockerfile                      # Containerização
└── docker-compose.yml              # Orquestração
```

---

### Aprovações

| Participante | Assinatura | Data |
|---|---|---|
|  | | |
|  | | |

---

*Relatório Final — Página 1 de 2*

*PMO Escritório de Projetos — http://escritoriodeprojetos.com.br*
