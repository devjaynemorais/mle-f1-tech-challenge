# KPI — Framework de Métricas para Experimentação de Machine Learning

## Objetivo

Este documento define o framework de métricas técnicas utilizado no processo de experimentação de modelos preditivos, com foco em problemas de classificação desbalanceada (ex.: churn). O objetivo é garantir comparabilidade entre experimentos, rastreabilidade no MLflow e alinhamento entre performance estatística e impacto de negócio.

---

## 1. Métricas Principais (Model Selection)

As métricas principais serão utilizadas como referência primária para comparação entre modelos durante a etapa de experimentação.

## 1.1 PR AUC (Precision-Recall Area Under Curve)

### Definição

Mede a área sob a curva Precision x Recall ao longo de múltiplos thresholds.

### O que captura

* Capacidade do modelo em identificar a classe positiva mantendo qualidade nas previsões positivas.
* Qualidade do ranking dos casos com maior probabilidade de churn.
* Trade-off entre cobertura e desperdício operacional.

### Relevância para o negócio

Em campanhas de retenção, falsos positivos geram custo. O PR AUC é especialmente relevante em bases desbalanceadas, pois penaliza modelos que capturam churners com baixa precisão.

### Diretriz

Principal KPI técnico para datasets com churn minoritário.

---

## 1.2 ROC AUC (Receiver Operating Characteristic Area Under Curve)

### Definição

Mede a área sob a curva TPR x FPR ao longo dos thresholds.

### O que captura

* Capacidade global de separação entre churners e não churners.
* Probabilidade de ranquear um churner acima de um não churner.

### Relevância para o negócio

Útil para comparar capacidade discriminativa geral entre modelos. Complementa o PR AUC.

### Diretriz

KPI técnico secundário-principal para benchmarking entre algoritmos.

---

## 2. Métricas Secundárias (Threshold / Operação)

Estas métricas serão usadas para interpretação adicional, tuning de threshold e análise operacional.

## 2.1 Recall

### Definição

Proporção de churners reais corretamente identificados.

### O que captura

* Cobertura da campanha.
* Quantos clientes em risco foram encontrados.

### Relevância para o negócio

Baixo recall implica perda de oportunidades de retenção.

---

## 2.2 F1-Score

### Definição

Média harmônica entre Precision e Recall.

### O que captura

* Equilíbrio entre cobertura e assertividade.
* Performance no threshold atual.

### Relevância para o negócio

Útil quando se deseja balancear custo de abordagem e captura de churners.

---

## 2.3 Fit Time

### Definição

Tempo de treinamento do modelo.

### O que captura

* Custo computacional.
* Eficiência de experimentação.
* Escalabilidade operacional.

### Relevância para o negócio

Modelos marginalmente melhores, porém muito lentos, podem não compensar operacionalmente.

---

## 2.4 Score Time / Predict Time

### Definição

Tempo necessário para gerar predições.

### O que captura

* Latência de inferência.
* Viabilidade para scoring em lote ou tempo real.

### Relevância para o negócio

Importante para janelas operacionais curtas e aplicações online.

---

## 3. Métricas Visuais e Diagnóstico

Estas análises devem ser geradas e registradas como artefatos no MLflow a cada experimento relevante.

## 3.1 Curva ROC

* Sensibilidade vs taxa de falso positivo.
* Diagnóstico global de discriminação.

## 3.2 Curva Precision-Recall

* Mais informativa em classes desbalanceadas.
* Avalia performance na classe churn.

## 3.3 Learning Curves

* Performance treino vs validação por tamanho amostral.
* Diagnóstico de underfitting / overfitting.

## 3.4 Validation Curves

* Sensibilidade da performance a hiperparâmetros.
* Apoio em tuning controlado.

## 3.5 Feature Importance

* Drivers principais do modelo.
* Interpretabilidade e suporte ao negócio.
* Pode ser nativa (árvores) ou via permutation importance / SHAP.

---

## 4. Hierarquia de Decisão Durante Experimentação

## Fase 1 — Triagem de Modelos

1. PR AUC
2. ROC AUC
3. Fit Time

## Fase 2 — Ajuste Operacional

1. Recall
2. F1-Score
3. Precision em thresholds específicos

## Fase 3 — Interpretação e Robustez

1. Curvas diagnósticas
2. Importances
3. Stability checks

---

## 5. Logging no MLflow

Cada experimento deverá registrar:

### Metrics

* pr_auc
  n- roc_auc
* recall
* f1_score
* fit_time
* score_time

### Params

* algoritmo
* hiperparâmetros
* features habilitadas
* versão dataset

### Artifacts

* roc_curve.png
* pr_curve.png
* learning_curve.png
* validation_curve.png
* feature_importance.png
* confusion_matrix.png

---

## 6. Diretriz Final

Durante experimentação, priorizar métricas threshold-independent (PR AUC / ROC AUC). Durante decisão operacional, complementar com métricas threshold-dependent (Recall, F1, Precision) e posteriormente métricas econômicas (ROI, revenue saved, churn evitado).

---

## 7. M?tricas Econ?micas Utilizadas

Esta se??o define as m?tricas econ?micas adotadas para avaliar campanhas de reten??o baseadas nas previs?es do modelo. O objetivo ? traduzir a matriz de confus?o em impacto financeiro, separando valor recuperado, valor perdido, custo operacional, desperd?cio com falsos positivos e retorno sobre o investimento.

### 7.1 Premissas

As defini??es abaixo assumem:

* a campanha ? acionada para todos os clientes preditos como churn;
* `TP + FP` representa o total de clientes acionados;
* `p_i` representa a probabilidade prevista de churn para o cliente `i`;
* `CLTV_i` representa o valor do cliente `i`;
* a taxa de reten??o da campanha representa a propor??o esperada de churners efetivamente retidos ap?s o acionamento;
* o custo operacional da campanha ? modelado como um **custo por acionamento**;
* o cen?rio base adota `custo_acionamento = 50` e `taxa_de_reten??o = 0.10`.

### 7.2 Valor em Risco Recuper?vel

**Defini??o**

Valor financeiro em risco dentro do subconjunto de churners corretamente identificados pelo modelo.

**F?rmula**

`VR = ? (p_i x CLTV_i), para i em TP`

**Interpreta??o**

Representa o montante potencialmente recuper?vel entre os clientes que o modelo decidiu acionar e que de fato pertencem ? classe churn.

### 7.3 Valor Recuperado Esperado

**Defini??o**

Valor esperado recuperado pela campanha ap?s considerar a taxa de reten??o.

**F?rmula**

`Vrec = VR x taxa_de_reten??o`

**Interpreta??o**

Nem todo churner acionado ser? salvo. Por isso, o valor recuperado esperado corresponde apenas ? fra??o do valor em risco que, em m?dia, a campanha consegue preservar.

### 7.4 Valor Perdido

**Defini??o**

Valor esperado perdido em churners reais que ficaram fora da campanha por erro do modelo.

**F?rmula**

`VP = ? (p_i x CLTV_i), para i em FN`

**Interpreta??o**

Corresponde ? oportunidade perdida causada por falsos negativos: clientes que churnariam, mas n?o foram acionados porque o modelo os classificou como n?o churn.

### 7.5 Custo por Acionamento

**Defini??o**

Custo unit?rio assumido para abordar um cliente dentro da campanha de reten??o.

**F?rmula**

`CMCA = custo_acionamento`

**Interpreta??o**

O custo m?dio por cliente acionado deixa de ser um rateio de budget fixo e passa a representar um custo operacional constante por contato.

### 7.6 Custo Total da Campanha

**Defini??o**

Investimento total necess?rio para executar a campanha no threshold analisado.

**F?rmula**

`custo_total_campanha = (TP + FP) x custo_acionamento`

**Interpreta??o**

Quanto mais clientes o modelo decide acionar, maior o custo total da campanha. Isso introduz um trade-off econ?mico expl?cito entre cobertura e desperd?cio operacional.

### 7.7 Valor Desperdi?ado com Falsos Positivos

**Defini??o**

Parcela do investimento da campanha consumida com clientes acionados desnecessariamente.

**F?rmula**

`VD = FP x custo_acionamento`

**Interpreta??o**

Mede o desperd?cio operacional gerado por falsos positivos: clientes que receberam esfor?o de reten??o, mas que n?o churnariam de qualquer forma.

### 7.8 Impacto Econ?mico L?quido

**Defini??o**

Saldo econ?mico da estrat?gia considerando valor recuperado, valor perdido e custo total de acionamento.

**F?rmula**

`IEL = Vrec - VP - custo_total_campanha`

**Interpreta??o**

Esta m?trica mostra se a pol?tica de acionamento induzida pelo modelo est? protegendo valor suficiente para compensar tanto a omiss?o de churners quanto o custo operacional da campanha.

### 7.9 ROI da Campanha

**Defini??o**

Retorno sobre o investimento total da campanha, considerando o valor recuperado, a perda por falsos negativos e o custo total de acionamento.

**F?rmula**

`ROI = (Vrec - VP - custo_total_campanha) / custo_total_campanha`

**Interpreta??o**

O ROI responde se, depois de descontar o custo integral de acionar os clientes previstos como churn, a estrat?gia baseada no modelo ainda gera retorno financeiro positivo.

### 7.10 Papel de Cada C?lula da Matriz de Confus?o

* `TP`: concentram o valor recuper?vel e alimentam `VR` e `Vrec`.
* `FP`: aumentam o custo total da campanha e alimentam `VD`.
* `FN`: representam oportunidade perdida e alimentam `VP`.
* `TN`: n?o geram recupera??o nem custo direto na formula??o atual.

### 7.11 Uso Recomendado no Notebook de Modelo Final

No notebook de consolida??o do modelo MVP, recomenda-se:

* comparar os modelos no holdout com base em `Vrec`, `VP`, `VD`, `IEL`, `ROI` e `custo_total_campanha`;
* analisar o efeito do threshold padr?o (`0.5`) versus threshold otimizado;
* testar sensibilidade do resultado econ?mico a diferentes premissas de taxa de reten??o e custo por acionamento.
