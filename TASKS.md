# Ludex: Quadro de Tarefas

Atualizado em 2026-06-21.

## Legenda

- ⏳ Pendente
- 🏗️ Em progresso
- ✅ Concluido
- ❌ Cancelado

## Fase 1: MVP 1 - concluido

### Dados e infraestrutura

- [✅] Estrutura de pastas e Docker inicial.
- [✅] Sourcing de dados migrado para RAWG.
- [✅] Pipeline de limpeza e processamento em `src/data/download_and_prep.py`.
- [✅] Fusao RAWG + IGDB em `src/data/super_dataset_merge.py`.
- [✅] Enriquecimento visual com imagens Steam em `src/data/add_images.py`.
- [✅] Contrato minimo de dados validado em `src/data/load_data.py` e no app.

### Recomendacao e interface

- [✅] App Streamlit em `app/streamlit_app.py`.
- [✅] Engine TF-IDF content-based.
- [✅] Modelo hibrido com conteudo, busca/opiniao e qualidade da comunidade.
- [✅] Perfil de gosto com multiplos jogos de referencia.
- [✅] Cards com imagem, score, explicacao e links externos.
- [✅] Busca textual por intencao reativada no fluxo principal.
- [✅] Cache TF-IDF persistente conectado ao app quando valido.
- [✅] Limpeza de tags ruidosas e hints de mecanica no corpus TF-IDF.
- [✅] UI limpa por padrao: detalhes tecnicos do ranking ficam recolhidos.

## Correcoes prioritarias pos-apresentacao

- [✅] Criar benchmark simples de qualidade com consultas e referencias fixas em `scripts/diagnostic/benchmark_recommendations.py`.
- [✅] Registrar top 5, latencia e explicacao para cada cenario em `docs/benchmark_latest.txt`.
- [⏳] Ajustar pesos do ranking com base no benchmark, nao apenas por intuicao.
- [⏳] Melhorar `review_keywords` com dados reais ou heuristicas auditaveis.
- [⏳] Resolver casos conceituais fracos do TF-IDF, especialmente `Minecraft Dungeons` -> ARPGs/looter/dungeon crawlers.
- [⏳] Transformar scripts soltos de diagnostico em testes automatizados.
- [✅] Benchmark inclui secao do Assistente Ludex com top 5 recuperado via RAG, fallback offline e latencia.
- [⏳] Documentar resultados antes/depois em `docs/`.

## Fase 2: MVP 2

## Fase 2: Reranking Semântico, Agente RAG Autônomo e Fine-Tuning (Concluído)

**Objetivo Atingido:** Melhorada a qualidade percebida (especialmente em buscas conceituais) usando um ChatBot Inteligente (Ollama/Llama 3) treinado sob medida (Fine-Tuning com Unsloth/LoRA) que pesquisa no motor do próprio aplicativo, abandonando a dependência do Amazon Bedrock e de custos em nuvem.

- [x] **Criar baseline de qualidade:**
  - Extraído `docs/baseline_v1.txt` com resultados do MVP (TF-IDF).
- [x] **Substituir o Motor Nuvem (Bedrock) por IA Local (Ollama):**
  - Módulo `OllamaExplainer` usa `LUDEX_OLLAMA_URL` e `LUDEX_OLLAMA_MODEL`, com defaults `http://localhost:11434` e `ludex-assistant:latest`.
  - Health check valida `/api/tags` e confirma se o modelo esperado existe.
  - Abordagem offline-first validada. O sistema não exige credenciais externas nem Ollama para abrir.
- [x] **Geração de justificativas com LLM Autônomo (RAG):**
  - O ChatBot não é mais passivo. Ele recebe a string de busca do usuário, junta com o histórico e dispara uma varredura real usando o `HybridRecommender`.
  - Injeção contextual das tags e gêneros, sem vazamento da sinopse em inglês.
  - Quando Ollama/modelo está indisponível, responde por fallback determinístico com top recomendações do ranking local.
- [x] **Fine-Tuning Local na RTX 4070 (Unsloth + LoRA):**
  - Script `generate_llm_dataset.py` gerando perguntas e respostas sintéticas a partir do `games.csv`.
  - Script `train_unsloth.py` otimizado para GPUs de 12GB rodando Llama 3 em 4-bit.
  - Modelo `ludex-assistant:latest` exportado nativamente para o Ollama.
- [x] **Integração na UI Premium (Streamlit):**
  - Design limpo e minimalista, focando no usuário final.
  - Excluída a matemática bruta e os cálculos percentuais zerados.
  - Emojis nativos removidos para garantir um visual elegante e unificado.
  - O app foi dividido em Abas limpas: `Vitrine` e `Assistente Ludex`.
  - Hero duplicado removido da aba `Vitrine`; status do assistente mostra Ollama ativo ou modo local.

- [x] **Higiene de artefatos locais:**
  - `.gitignore` ignora pesos GGUF, checkpoints, outputs de treino, caches Unsloth/Hugging Face e datasets gerados.
  - Mantido manifest pequeno em `ludex-llama3-finetuned_gguf/model_manifest.json` com comando de recriação.

## Notas de coordenacao

- O app deve continuar funcionando localmente sem AWS.
- O app deve continuar funcionando mesmo sem Ollama; nessa condição o Assistente usa fallback local.
- Nao remover o TF-IDF ate a fase 2 provar melhora objetiva.
- Antes de qualquer demo, rodar `python scripts/warmup_cache.py`.
- Consultar `AGENTS.md` e `docs/architecture.md` antes de mexer em arquitetura.
