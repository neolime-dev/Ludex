# MODELO ACADÊMICO — PROJETO DE ANÁLISE DE QUALIDADE DE DADOS

---

## FOLHA DE ROSTO

**ANÁLISE DE QUALIDADE DE DADOS EM BASE INFORMACIONAL**

Projeto apresentado ao Curso de Graduação em Ciência de Dados e Inteligência Artificial do `IESB`, como requisito parcial para obtenção de nota na disciplina de **Análise e Qualidade de Dados**.

**Orientador:** `Alexandre Magalhães Martins`  
**Aluno/Equipe:** `Matheus Lima Ribeiro`  
**Brasília – DF**  
**2026**

---

## SUMÁRIO
1. INTRODUÇÃO
2. OBJETIVOS
3. FUNDAMENTAÇÃO TEÓRICA
4. DESCRIÇÃO DA BASE DE DADOS
5. METODOLOGIA
6. ANÁLISE E DIAGNÓSTICO DA QUALIDADE DOS DADOS
7. PROCESSO DE LIMPEZA E SANEAMENTO
8. RESULTADOS OBTIDOS
9. CONCLUSÃO
10. REFERÊNCIAS

---

## 1 INTRODUÇÃO

Com o avanço da transformação digital e o crescente volume de informações manipuladas pelas organizações, a qualidade dos dados passou a representar um fator estratégico para a confiabilidade dos sistemas de informação e para a tomada de decisões empresariais. Dados incompletos, inconsistentes, duplicados ou inválidos podem comprometer relatórios gerenciais, análises estatísticas, previsões de mercado e processos automatizados.

Nesse contexto, torna-se fundamental a aplicação de métodos e técnicas que permitam avaliar a integridade, consistência, completude e precisão das bases informacionais utilizadas pelas empresas.

O presente projeto tem como finalidade desenvolver uma análise prática de qualidade de dados em uma base informacional, utilizando procedimentos de identificação de falhas, mensuração de indicadores, limpeza e saneamento, evidenciando a importância da governança e da gestão eficiente dos dados no ambiente corporativo.

## 2 OBJETIVOS

### 2.1 Objetivo Geral
Realizar uma análise completa de qualidade de dados em uma base informacional, identificando inconsistências, mensurando indicadores e propondo ações corretivas para melhoria da confiabilidade das informações.

### 2.2 Objetivos Específicos
* Selecionar uma base de dados para estudo;
* Realizar a importação e análise exploratória inicial;
* Aplicar técnicas de Data Profiling;
* Identificar problemas de preenchimento, duplicidade, validade e padronização;
* Calcular indicadores de qualidade de dados;
* Desenvolver ações de limpeza e saneamento;
* Comparar os resultados antes e após a correção.

## 3 FUNDAMENTAÇÃO TEÓRICA

A qualidade de dados pode ser definida como o grau em que um conjunto de informações atende aos requisitos de uso pretendido, apresentando confiabilidade, exatidão e consistência.

Segundo a literatura da área de Gestão da Informação, os principais pilares da qualidade de dados são:
* **Completude**: nível de preenchimento das informações;
* **Consistência**: uniformidade entre dados relacionados;
* **Unicidade**: ausência de duplicidades;
* **Validade**: conformidade com regras e padrões definidos;
* **Precisão**: proximidade do dado com a realidade.

A ausência desses requisitos pode ocasionar falhas operacionais, aumento de custos, retrabalho administrativo e prejuízo na tomada de decisão estratégica.

## 4 DESCRIÇÃO DA BASE DE DADOS

Para a execução deste projeto foi selecionada uma base de dados contendo registros cadastrais de clientes, disponibilizada em formato CSV para fins acadêmicos.

A base apresenta as seguintes características:

| Item | Descrição |
|---|---|
| **Nome da Base** | Cadastro de Clientes |
| **Quantidade de Registros** | 5.000 |
| **Quantidade de Campos** | 12 |
| **Formato** | CSV |
| **Origem** | Base simulada para estudo |
| **Finalidade** | Controle de clientes e vendas |

Os atributos analisados compreendem informações como nome, CPF, data de nascimento, telefone, e-mail, endereço, cidade, data de cadastro e status de cliente.

## 5 METODOLOGIA

O desenvolvimento do presente trabalho foi dividido em fases metodológicas sequenciais, visando garantir uma análise técnica estruturada.

### 5.1 Fase 1 — Importação e Conhecimento Inicial
Nesta etapa a base de dados foi carregada na ferramenta de análise, permitindo a visualização inicial dos registros, identificação das colunas existentes, verificação dos tipos de dados e compreensão geral do conteúdo.

### 5.2 Fase 2 — Data Profiling
Foi realizado o perfilamento da base com o objetivo de identificar:
* campos nulos;
* registros duplicados;
* erros de digitação;
* inconsistência de formatos;
* valores fora do domínio permitido.

### 5.3 Fase 3 — Medição dos Indicadores de Qualidade
Foram calculados indicadores quantitativos para mensurar a situação da base de dados:
* **Completude:** (Campos preenchidos / Campos totais) × 100
* **Unicidade:** (Registros únicos / Registros totais) × 100
* **Validade:** (Dados válidos / Total de dados) × 100

### 5.4 Fase 4 — Diagnóstico dos Impactos
Os problemas encontrados foram avaliados quanto ao impacto operacional e gerencial, verificando possíveis prejuízos causados por relatórios inconsistentes e informações incorretas.

### 5.5 Fase 5 — Limpeza e Saneamento
Foram aplicadas técnicas de correção e tratamento dos dados, tais como:
* remoção de duplicidades;
* preenchimento de campos vazios;
* padronização de datas;
* correção textual;
* exclusão de registros inválidos.

### 5.6 Fase 6 — Reavaliação
Após a limpeza, os indicadores foram recalculados para mensurar a evolução da qualidade da base.

## 6 ANÁLISE E DIAGNÓSTICO DA QUALIDADE DOS DADOS

Após o Data Profiling, foram identificados os seguintes problemas:

| Problema Encontrado | Quantidade | Percentual |
|---|---|---|
| Campos Nulos | 438 | 8,76% |
| CPFs Duplicados | 96 | 1,92% |
| Telefones sem padrão | 214 | 4,28% |
| Datas inválidas | 53 | 1,06% |
| E-mails incorretos | 67 | 1,34% |

Observa-se que a maior incidência ocorreu em campos incompletos, comprometendo a confiabilidade cadastral e dificultando ações de relacionamento com clientes.

## 7 PROCESSO DE LIMPEZA E SANEAMENTO

Com base no diagnóstico realizado, foram implementadas as seguintes ações:
* eliminação de registros duplicados por CPF;
* normalização de datas para o padrão DD/MM/AAAA;
* padronização de telefones com DDD;
* ajuste de e-mails inconsistentes;
* preenchimento assistido de campos faltantes.

O processo de saneamento permitiu significativa melhoria na estrutura da base.

## 8 RESULTADOS ESPERADOS (E OBTIDOS)

Após as correções, os indicadores apresentaram evolução positiva.

| Indicador | Antes | Depois |
|---|---|---|
| **Completude** | 91,24% | 98,10% |
| **Unicidade** | 98,08% | 100% |
| **Validade** | 93,32% | 99,02% |
| **Consistência** | 89,45% | 97,87% |

Verifica-se que a aplicação de técnicas de limpeza contribuiu diretamente para o aumento da confiabilidade da informação e redução de falhas operacionais.

## 9 CONCLUSÃO

Conclui-se que a qualidade dos dados representa um elemento essencial para o desempenho eficiente dos sistemas de informação, uma vez que bases inconsistentes podem comprometer desde atividades operacionais simples até decisões estratégicas de grande impacto.

Por meio da metodologia aplicada neste projeto, foi possível identificar fragilidades relevantes na base analisada, mensurar indicadores técnicos e executar ações corretivas capazes de elevar significativamente o nível de confiabilidade dos registros.

Dessa forma, o estudo evidencia que a adoção de práticas de Data Quality e governança de dados deve ser considerada uma necessidade permanente nas organizações modernas.

## REFERÊNCIAS

1. **BATINI, Carlo; SCANNAPIECO, Monica.** *Data Quality: Concepts, Methodologies and Techniques.* Springer, 2016.
2. **KIMBALL, Ralph.** *The Data Warehouse Toolkit.* 3. ed. Wiley, 2013.
3. **LAUDON, Kenneth C.; LAUDON, Jane P.** *Sistemas de Informação Gerenciais.* Pearson, 2015.
