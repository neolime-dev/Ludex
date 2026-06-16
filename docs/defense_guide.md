# 🛡️ Guia de Defesa da Banca: Projeto Ludex

Este guia contém argumentos técnicos para defender o Ludex contra perguntas difíceis do professor.

## 1. "Por que usar TF-IDF se existem modelos como BERT ou Claude?"
**Resposta Estratégica:** 
"Para este MVP, o TF-IDF com N-Grams foi escolhido por três motivos: 
1. **Performance e Baixa Latência:** Ele roda 100% localmente sem necessidade de GPUs caras ou APIs pagas, permitindo que a busca seja instantânea. 
2. **Explainable AI (XAI):** Diferente de um modelo 'caixa preta' de Deep Learning, o TF-IDF nos permite mostrar exatamente qual palavra (ex: 'Dungeon Crawler') causou a recomendação, o que é fundamental para a confiança do usuário. 
3. **Calibração de Pesos:** Conseguimos dar pesos manuais para Tags (10x) e Gêneros (5x), corrigindo o viés de modelos pré-treinados que nem sempre entendem o nicho de games."

## 2. "Como vocês garantem a qualidade dos dados vindo de fontes diferentes?"
**Resposta Estratégica:**
"Desenvolvemos um **Super Dataset** via pipeline de Engenharia de Dados. Usamos o RAWG como fonte mestre para Tags da Comunidade (que são mais precisas que metadados de loja) e o IGDB para enriquecer as descrições textuais. Aplicamos filtros de sanidade (Drop Duplicates, NSFW Filter e Relevância por Rating) para garantir que apenas os 40.000 jogos mais qualificados entrassem no motor."

## 3. "O que acontece se eu buscar por um termo que não existe (Out-of-Vocabulary)?"
**Resposta Estratégica:**
"Nosso motor Híbrido resolve isso com uma estratégia de **Fallback Multinível**. Se o TF-IDF não encontrar similaridade semântica, o sistema entra com o score de **Popularidade e Sentimento da Comunidade**, garantindo que o usuário nunca receba uma página vazia, mas sim os jogos mais aclamados do momento."

## 4. "Por que focar em Bigramas e Trigramas?"
**Resposta Estratégica:**
"No domínio de games, o contexto é tudo. A palavra 'Open' e a palavra 'World' separadas têm significados genéricos. Juntas como um **Bigrama**, elas definem uma categoria inteira. Usar `ngram_range=(1, 3)` no TfidfVectorizer foi o que permitiu ao Ludex entender conceitos complexos como 'Turn Based RPG' ou 'Fast Paced Action'."

---
*Dica do Gemini: Se o professor elogiar a interface, mencione que o projeto segue padrões de design corporativo com foco em UX centrada na busca.*
