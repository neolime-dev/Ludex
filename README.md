# 🎮 Ludex: Sistema de Recomendação de Jogos Multiestratégia

O Ludex é um motor de recomendação de jogos de alta fidelidade que utiliza Processamento de Linguagem Natural (NLP) avançado para conectar jogadores a novos títulos baseados em mecânicas, temas e sentimentos da comunidade.

## 🚀 Diferenciais Técnicos (MVP 1)
- **Super Dataset:** Fusão proprietária de dados do **RAWG** (tags de mecânica) e **IGDB** (descrições ricas), totalizando +39.000 títulos.
- **Motor NLP Calibrado:** Utilização de **TF-IDF com N-Grams** (Bigramas/Trigramas) e calibração de pesos (10x para Tags, 5x para Gêneros).
- **Perfil de Gosto Composto:** Suporte a múltiplos jogos de referência através de cálculo de centroide vetorial.
- **Interface Premium:** Design customizado "Steam-Style" com ícones Lucide e modo escuro profissional.
- **Explicabilidade (Explainable AI):** Demonstração matemática dos termos que causaram cada recomendação.

## 🛠️ Como Executar (Docker)

1. Clone o repositório:
   ```bash
   git clone https://github.com/neolime-dev/Ludex.git
   cd Ludex
   ```

2. Suba o ambiente:
   ```bash
   docker-compose up -d
   ```

3. Acesse no navegador:
   `http://localhost:8501`

---
## 🎯 Próximos Passos (Fase 2)
- Integração com **Amazon Bedrock** (Amazon Titan Embeddings).
- Justificativas textuais geradas por LLM (Claude 3).
- Cache semântico persistente.
