# Ludex: Quadro de Tarefas (TASKS.md)

## 📋 Status Legenda
- ⏳ Pendente
- 🏗️ Em Progresso
- ✅ Concluído
- ❌ Cancelado

---

## 🎯 Fase 1: MVP 1 (Prazo: Amanhã 16/06) - CONCLUÍDO ✅
### Dados & Infra (Responsável: Gemini)
- [✅] Estrutura de pastas e Docker inicial
- [✅] Sourcing de dados (Migrado para RAWG - 40k jogos)
- [✅] Pipeline de limpeza e processamento (download_and_prep.py)
- [✅] Notebook de análise exploratória (01_model_evaluation.ipynb)
- [✅] Fusão de Datasets: RAWG + IGDB (Super Dataset)
- [✅] Configuração de Remote GitHub (neolime-dev/Ludex)

### Interface & Estética (Responsável: Codex)
- [✅] Esqueleto Streamlit
- [✅] Engine TF-IDF (Content-Based)
- [✅] Modelo Híbrido (hybrid.py)
- [✅] Refino de UX/UI (Layout Steam-style Finalizado)
- [✅] Polimento Final (Commit de Ouro 4fc8097)
- [✅] Redesign Premium (Cards high-end, painel customizado e CSS avancado)

### NLP Acadêmico (Responsável: Colaborador Externo / Amigo)
- [🏗️] **Missão B:** Implementar Explicação de Recomendações (Keywords TF-IDF)
- [⏳] Integrar explicações visuais no Streamlit

---

## 🎯 Fase 2: MVP 2 (Prazo: 22/06)
### Integração Amazon Bedrock
- [⏳] Setup de credenciais AWS Boto3
- [⏳] Semantic Embeddings (Titan) e Cache
- [⏳] Geração de justificativas via LLM (Claude 3)

---

## 📝 Notas de Coordenação
- **MVP 1 Entrega:** Versão 1.0.0 estável no ar.
- **Destaque:** Diablo III agora é recomendado para Minecraft Dungeons (Pesos calibrados).
