# Demandas Gemini -> Codex

## Demanda #4: NLP de Opinião e Integração de Reviews (Foco Acadêmico)
- **Status:** Aberta
- **Prioridade:** Crítica (Diferencial do Projeto)
- **Data:** 15/06/2026

### Descrição:
O usuário quer que as **reviews e opiniões** sejam fundamentais na recomendação. O objetivo é permitir que o usuário busque por termos subjetivos (ex: "jogo emocionante", "frustrante", "viciante") e que o sistema use o sentimento das reviews como peso.

**Tarefas:**
1. **Engine de Busca Opinativa:** O `TfidfVectorizer` deve incluir agora as reviews agregadas (ou as keywords de sentimento que eu vou extrair no pipeline de dados).
2. **Sentiment Weighting:** Crie uma função no `hybrid.py` que dê um bônus de score para jogos com sentimento predominantemente positivo nas reviews.
3. **UI de Sentimento:** No card do jogo (Steam-style), adicione um "Resumo da Comunidade" (ex: "A maioria dos jogadores acha este jogo 'Desafiador' e 'Imersivo'").

### Contrato de Dados Atualizado:
Vou fornecer em `data/processed/games.csv` a coluna `review_keywords` (termos mais citados em reviews) e `sentiment_score`.

### Próximos Passos:
- Atualize o modelo híbrido para: `Score = (0.4 * Conteúdo) + (0.3 * Busca Opinativa) + (0.3 * Popularidade/Sentimento)`.
