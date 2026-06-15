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
- [✅] Sourcing de dados (Migrado para RAWG - 40k jogos)
- [✅] Pipeline de limpeza e processamento (download_and_prep.py)
- [✅] Notebook de análise exploratória (01_model_evaluation.ipynb)
- [🏗️] Fusão de Datasets: RAWG + IGDB (Análise de viabilidade)

### Interface & Estética (Responsável: Codex)
- [✅] Esqueleto Streamlit
- [✅] Engine TF-IDF (Content-Based)
- [✅] Modelo Híbrido (hybrid.py)
- [✅] Refino de UX/UI (Layout Steam-style e Cards)
- [✅] Polimento Final (Ajustes de 11% de cota)

### NLP Acadêmico (Responsável: Colaborador Externo / Amigo)
- [⏳] **Missão B:** Implementar Explicação de Recomendações (Keywords TF-IDF)
- [⏳] Integrar explicações visuais no Streamlit

---

## 🎯 Fase 2: MVP 2 (Prazo: 22/06)
### Integração Amazon Bedrock
- [⏳] Setup de credenciais AWS Boto3
- [⏳] Semantic Embeddings (Titan) e Cache
- [⏳] Geração de justificativas via LLM (Claude 3)

---

## 📝 Notas de Coordenação
- **Fluxo de Trabalho:** O Amigo deve trabalhar na branch `feature/nlp-explain`.
- **Merge:** Gemini revisará os Pull Requests do colaborador externo.
