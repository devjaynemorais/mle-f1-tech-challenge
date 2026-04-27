# Metodologia de Experimentação para Modelos de Churn

## Objetivo

Estabelecer um fluxo estruturado e reprodutível para desenvolvimento de modelos preditivos de churn, equilibrando performance estatística, impacto financeiro e viabilidade operacional.

## Princípios Norteadores
1.	Começar simples e evoluir com evidência.
2.	Separar predição de decisão de negócio.
3.	Comparar modelos sob mesmas condições experimentais.
4.	Usar métricas técnicas e econômicas em conjunto.
5.	Reduzir complexidade antes de aprofundar simulações.

## Protocolo de Validação: 

**Validação Cruzada Estratificada - `StratifiedKFold`**

**Motivo**: Base moderadamente desbalanceada, optando por não considerar técnicas de balanceamento nesse primeiro momento.

## Visão geral do Workflow de Experimentação
### Fase 1 — Baseline Inicial (MVP)
#### Objetivo

Realizar uma exploração técnica rápida para avaliar o poder preditivo inicial do dataset, construindo um modelo de regressão logística e dummy como baseline inicial e comparando com outros diferentes modelos (MLP, RandomForest, XGBoost e DecisionTree).

#### **Principais KPI's técnicos**
`PR AUC` e `ROC AUC` seguem como métricas principais por fornecerem uma visão mais global do desempenho do modelo sem depender tanto de um threshold que será otimizado futuramente. 

>**IMPORTANTE**: consultar `KPI.md` para uma melhor descrição das métricas.

**Benchmark para PR-AUC**

| PR AUC / Base rate | Interpretação    |
| ------------------ | ---------------- |
| 1.0x               | Aleatório        |
| 1.2x               | Fraco            |
| 1.5x               | Útil             |
| 2.0x               | Bom              |
| 3.0x               | Muito forte      |
| 4x+                | Excelente / raro |


**Benchmark para ROC-AUC**
| ROC AUC     | Interpretação                   |
| ----------- | ------------------------------- |
| 0.50        | Aleatório                       |
| 0.55 – 0.60 | Fraco                           |
| 0.60 – 0.70 | Básico / MVP                    |
| 0.70 – 0.80 | Bom                             |
| 0.80 – 0.90 | Muito bom                       |
| > 0.90      | Excelente / raro em dados reais |
| 1.00        | Perfeito (suspeito em produção) |


#### **KPI's Secundários (possível desempate)**
- Recall
- Precision
- F1-Score
- Tempo de treino 
- Estabilidade CV 
- Overfitting/underfitting (learning curve, validation curves)
- Interpretabilidade básica (feature importance, coeficientes, SHAP)

#### **O que queremos responder:**

- Há sinal preditivo nas features atuais? 
- O dataset é minimamente útil? 
- Problema linear ou não linear? 
- As árvores dominam? 
- Overfitting aparece cedo? 
- Qual benchmark mínimo aceitável para experimentação?


#### **Saída:**
- **No Jupyter Notebook:** Tabela Comparativa entre modelos contendo os KPI’s técnicos como output + visualizações gráficas (**learning curve**, **plot importance**, etc)

    Exemplo:

    | Modelo   | PR AUC | ROC AUC | Recall | Precision | F1-Score | Fit time | Score time | Estabilidade CV |
    |----------|--------|---------|--------|-----------|----------|----------|------------|-----------------|
    | Logistic | 0.41   | 0.72    | 0.63   | 0.53      | 0.57     | 0.0132   | 0.0124     | 3%              |
    | XGB      | 0.49   | 0.70    | 0.67   | 0.51      | 0.54     | 0.123    | 0.124      | 4%              |
    | MLP      | 0.48   | 0.75    | 0.69   | 0.55      | 0.59     | 0.204    | 0.124      | 2%              |

- **No MLFlow:** logging dos experimentos, métricas e artefatos (gráficos não nativos) no MLFlow.
  
  Exemplo: 
  - Experiment 1 - baseline_models
  - Experiment 2 - benchmark_models

#### **Conclusão:**



### Fase 2 — Feature Engineering
#### **Objetivo**
Adicionar poder preditivo ao modelo de forma controlada e avaliar nos modelos candidatos.

#### **Features Adicionadas**
- **Targeting Encoding na variável `City`**: Substitui a cidade pela taxa histórica de Churn daquela cidade. Mas precisa fazer em validação cruzada **out-of-fold** para evitar leakage.
- **Agrupamento de `Ternure`**: Durante a EDA ficou evidente que o churn é mais comum em clientes mais novos do que em clientes antigos. A ideia é agrupar Ternure em faixas e aplicar OrdinalEncoding mantendo o risco de acordo com a monotocidade. Mapping utilizado {'new': 2, 'mid': 1, 'loyal': 0}.
- **Transformação Logarítmica de `Ternure`**: A relação entre tempo de permanência do cliente e churn não segue uma relação linear.
- **digital_engagement_score**: é uma flag que somatiza todos os serviços digitais contratado pelo cliente (retorna um valor numérico de 0 a 4).
- **Aplicação de Ordinal Encoding na variável `Contract`**: A EDA revelou que a variável de tipo de contrato atua como um proxy de mitigação de churn. Contratos mensais são mais propensos a churn do que contratos anuais, que por sua vez são mais propensos ao churn do que contratos Bi-anuais.
- **Adição da Flag de Estabilidade Familiar**: Uma flag que sinaliza dependentes o que pode ser uma barreira natural de cancelamento do serviço.
- **Adição de Flag para Serviço de Fibra, sem Suporte de serviço**: proxy fortíssimo que aumenta muito os casos de churns devido a insatisfação com o serviço contratado.
- **Adição da variável `Churn_Score`**: variável que é um score de outro modelo interno da empresa.

#### **Estratégia de Seleção de Features Testadas**
- **SelectKBest**: `f_classif (test-t ANOVA)` para capturar variabilidade de forma isolada e `mutual_information (teste não paramétrico)` para capturar não linearidades e correlações.
- **RFE**: Eliminação recursiva, introduz custo computacional, porém retreina o modelo eliminando sempre a pior feature. Bom para capturar interações entre as features.
- **Model-Based Selection**: Aplicar regularizações nos modelos (L1) e deixar o modelo decidir quais features são mais relevantes. 

#### **Saída**
- **No Jupyter Notebook:** Tabela Comparativa entre modelos contendo os KPI’s técnicos como output + plot importances(subir como artefato no MLFlow para controle features selecionadas)

    Exemplo:

    | experimento  | PR AUC | ROC AUC | Recall | Precision | F1-Score | Fit time | Score time | Estabilidade CV |
    |----------|--------|---------|--------|-----------|----------|----------|------------|-----------------|
    | RegLog_selKBest_f_classif | 0.41   | 0.72    | 0.63   | 0.53      | 0.57     | 0.0132   | 0.0124     | 3%              |
    | RegLog_selKBest_MI     | 0.49   | 0.70    | 0.67   | 0.51      | 0.54     | 0.123    | 0.124      | 4%              |
    | RegLog_L1      | 0.48   | 0.75    | 0.69   | 0.55      | 0.59     | 0.204    | 0.124      | 2%              |
    | RegLog_RFE | 0.41   | 0.72    | 0.63   | 0.53      | 0.57     | 0.0132   | 0.0124     | 3%              |
    | XGB_SelKBest_MI      | 0.49   | 0.70    | 0.67   | 0.51      | 0.54     | 0.123    | 0.124      | 4%              |
    | MLP_SelKBest_f_classif      | 0.48   | 0.75    | 0.69   | 0.55      | 0.59     | 0.204    | 0.124      | 2%              | 
    | XGB_L1 | 0.41   | 0.72    | 0.63   | 0.53      | 0.57     | 0.0132   | 0.0124     | 3%              |
 

- **No MLFlow:** logging dos experimentos, métricas e artefatos (gráficos não nativos) no MLFlow + dataset versionado para cada experimento junto com as features selecianadas.
  
  Exemplo: 
  - Experiment 3 - benchmark_FE
    - RegLog_selKBest_f_classif
    - RegLog_selKBest_MI
    - RegLog_L1
    - RegLog_RFE
    - MLP_SelKBest_f_classif
    - MLP_SelKBest_MU
    - MLP_L1


#### **Conclusão**




### Registros no MLFlow

- Experiment 1 - baseline_models
- Experiment 2 - benchmark_models
- Experiment 3 - benchmark_FE
- Experiment 4 - benchmark_Economics
- Experiment 5 - tunning_best_model

### Estrutura de Notebooks Utilizada

- 02_experiment_fase01_baseline
- 03_experiment_fase02_FE
- 04_experiment_fase03_economic_analisys
- 05_experiment_fase04_tunning_best_model

### Boas Práticas Adotadas
- Não misturar feature engineering pesado com simulação econômica em todas as rodadas.
- Primeiro descubra qual família aprende melhor o problema. Depois descubra quais features maximizam essa família.
- Primeiro reduzir espaço técnico, depois aprofundar análise financeira.
- Realizar Tunning apenas no modelo campeão.
- Decidir o Modelo Campeão com base na Análise Econômica (melhor ROI). Nem sempre Melhor AUC retornará o melhor lucro.
- CLTV deve ser melhor usado como metadata de decisão do que como feature direta.
- Durante a Experimentação foram mantidas as mesmas condições em todos os experimentos (splits e premissas de negócio).

### Regras Gerais
- Para facilitar a filtragem dos modelos, os resultados foram ordenados na seguinte ordem: PR-AUC > ROC-AUC > Recall > F1-Score > Fit Time > Estabilidade.
- 