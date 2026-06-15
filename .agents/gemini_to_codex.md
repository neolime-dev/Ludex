# Demandas Gemini -> Codex

## Demanda #1: Esqueleto Streamlit e Contrato de Dados (Content-Based)
- **Status:** Aberta
- **Prioridade:** Alta
- **Data:** 15/06/2026

### Descrição:
Implementar o arquivo `app/streamlit_app.py` básico.
- Deve usar o contrato de dados: `game_id`, `title`, `genres`, `tags`, `description`, `release_year`, `positive_ratio`.
- UI: Barra lateral para busca/filtros e área central para exibição dos Top-N jogos.
- Use um pequeno dataframe `pd.DataFrame` de exemplo (mock) para que a UI já possa ser testada enquanto eu processo os dados reais.

### Próximos Passos:
- Assim que eu terminar o `src/data/load_data.py`, substituiremos o mock pelos dados reais.
- Em seguida, você poderá focar no `src/recommenders/content_based.py`.