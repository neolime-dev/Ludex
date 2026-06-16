# 🎤 Roteiro de Apresentação: Projeto Ludex (MVP 1)

Este roteiro foi estruturado para uma fala de **5 a 10 minutos**, focando nos pontos que os professores de NLP e Data Science mais valorizam.

---

## 1. Introdução e Problema (1 min)
*   **Abertura:** "Bom dia/noite, professor. Este é o Ludex, um sistema de recomendação multiestratégia focado no universo de games."
*   **O Problema:** "Sistemas de recomendação comuns sofrem de 'Frieza Semântica'. Se você gosta de *Minecraft Dungeons*, um algoritmo comum te daria outros jogos de 'blocos'. Mas o jogador quer a **mecânica** (Action RPG/Dungeon Crawler) e não a estética."
*   **O Objetivo:** "Criar um motor que entenda a 'alma' do jogo através do processamento de linguagem natural das tags e descrições."

---

## 2. Engenharia de Dados (2 min)
*   **Super Dataset:** "Não usamos apenas uma fonte. Fizemos a fusão (Merge) de dois grandes datasets: o **RAWG** (pelas tags ricas da comunidade) e o **IGDB** (pelas descrições detalhadas)."
*   **Filtros de Qualidade:** "Processamos +120 mil jogos e filtramos para os 40 mil mais relevantes, aplicando remoção de duplicatas e um filtro rigoroso de segurança (NSFW) para garantir uma vitrine profissional."
*   **Infraestrutura:** "O projeto é 100% containerizado via **Docker**, garantindo que o ambiente de execução seja idêntico em qualquer máquina."

---

## 3. O Motor de NLP (A Parte Técnica) (3 min)
*   **Vetorização:** "Utilizamos **TF-IDF com N-Grams (1, 2)**. Isso é crucial porque termos como 'Open World' ou 'Turn Based' precisam ser entendidos como conceitos únicos, e não palavras separadas."
*   **Calibração de Pesos (O Diferencial):** "Este é o 'pulo do gato': Aplicamos pesos diferenciados no corpus. As **Tags têm peso 10x** e os **Gêneros peso 5x** em relação à descrição comum. Isso força a IA a priorizar a jogabilidade sobre o título."
*   **Centroide Vetorial:** "Na função de 'Perfil de Gosto', o usuário seleciona vários jogos. O sistema calcula a **média vetorial (centroide)** dessas escolhas para encontrar o ponto exato de intersecção entre diferentes gostos."

---

## 4. Demonstração Prática (2 min)
*   **Visual:** "Adotamos uma interface 'Clean' inspirada no padrão SaaS moderno, fugindo do visual padrão de templates de IA."
*   **Teste Real:** (Faça a busca por *Minecraft: Dungeons*).
*   **O Triunfo:** "Vejam que, mesmo com nomes diferentes, o sistema recomenda *Diablo III* e *Torchlight*. Isso prova que nossa calibração de pesos em Tags de mecânica funcionou."
*   **Explainable AI:** "Cada card mostra o 'Porquê' da recomendação, expondo os termos TF-IDF que mais contribuíram para o score."

---

## 5. Conclusão e Fase 2 (1 min)
*   **MVP 1:** "Entregamos uma engine funcional, rápida (com sistema de cache persistente) e academicamente fundamentada."
*   **Próximos Passos:** "Na Fase 2, integraremos o **Amazon Bedrock** para substituir o TF-IDF por **Embeddings Semânticos densos** e usar LLMs para gerar justificativas em linguagem natural."

---

### 💡 Dicas de Ouro para as Perguntas:
1.  **Se perguntar sobre o "Efeito Fortnite":** Diga que recalibrou o motor para que a similaridade textual (70%) domine sobre a popularidade (15%).
2.  **Se perguntar sobre o Cache:** Mencione que o modelo é pré-treinado e salvo em `.pkl` para garantir latência zero na busca.
3.  **Se perguntar sobre Lemmatização:** Diga que o Tfidf do Scikit-learn com `strip_accents` e `stop_words` já resolve o ruído para este volume de dados.

---
*Boa sorte, Arquiteto! O Ludex está pronto para o show.* 🕹️🏆
