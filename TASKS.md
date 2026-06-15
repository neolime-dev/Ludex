# Ludex: Quadro de Tarefas (TASKS.md)

## 📋 Status Legenda
- ⏳ Pendente
- 🏗️ Em Progresso
- ✅ Concluído
- ❌ Cancelado

---

## 🎯 Fase 1: MVP 1 (Prazo: Amanhã 16/06)
### Dados & Infra (Responsável: Gemini)
- [✅] Estrutura de pastas e Docker inicial
- [✅] Sourcing de dados UCSD (32k jogos)
- [✅] Pipeline de limpeza e processamento (download_and_prep.py)
- [✅] Pipeline streaming de reviews (process_reviews.py)
- [✅] Notebook de análise exploratória (01_model_evaluation.ipynb)
- [⏳] Configuração de Remote GitHub (Aguardando URL)

### Core ML & Interface (Responsável: Codex)
- [✅] Esqueleto Streamlit
- [✅] Engine TF-IDF (Content-Based)
- [✅] Modelo Híbrido (hybrid.py)
- [✅] Refino de UX/UI (Cards e Layout Steam-style)
- [✅] Links externos nos cards (Steam + PCGamingWiki)
- [✅] Busca opinativa com reviews/sentimento (review_keywords + sentiment_score)
- [⏳] NLP Avançado: Lemmatization e N-Grams
- [🏗️] NLP Avançado: Extração de Keywords (Explicabilidade)

---

## 🎯 Fase 2: MVP 2 (Prazo: 22/06)
### Integração Amazon Bedrock
- [⏳] Setup de credenciais AWS Boto3
- [⏳] Semantic Embeddings (Titan) e Cache
- [⏳] Geração de justificativas via LLM (Claude 3)

---

## 📝 Notas de Coordenação
- **Comunicação:** Usar `.agents/` para contratos técnicos.
- **NLP Focus:** Reforçar técnicas acadêmicas para a apresentação de amanhã.
- **Contrato CSV atual:** colunas mínimas `game_id,title,genres,tags,description,release_year,positive_ratio`; opcionais usadas no app: `price,developer,publisher,url_store,url_ref,review_keywords,sentiment_score`.
- **UX:** App em dark mode Steam-style com paleta `#1b2838`, `#16202d`, `#66c0f4`; filtros avançados ficam na sidebar.
- **Score opinativo:** fórmula preparada para `0.4 * content_score + 0.3 * opinion_score + 0.3 * quality_score`, onde `quality_score` combina popularidade e `sentiment_score` quando disponível.
- **Reviews:** `src/data/process_reviews.py` baixa/processa `steam_reviews.json.gz` em streaming e gera `data/processed/reviews_agg.csv` com `review_keywords` e `sentiment_score`.
