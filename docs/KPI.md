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

## 7. Métricas Econômicas Utilizadas

Esta seção define as métricas econômicas adotadas para avaliar campanhas de retenção baseadas nas previsões do modelo. O objetivo é traduzir a matriz de confusão em impacto financeiro, separando valor recuperado, valor perdido, desperdício operacional e retorno sobre o investimento.

### 7.1 Premissas

As definições abaixo assumem:

* a campanha é acionada para todos os clientes preditos como churn;
* `TP + FP` representa o total de clientes acionados;
* `p_i` representa a probabilidade prevista de churn para o cliente `i`;
* `CLTV_i` representa o valor do cliente `i`;
* a taxa de retenção da campanha representa a proporção esperada de churners efetivamente retidos após o acionamento.

### 7.2 Valor em Risco Recuperável

**Definição**

Valor financeiro em risco dentro do subconjunto de churners corretamente identificados pelo modelo.

**Fórmula**

`VR = Σ (p_i x CLTV_i), para i em TP`

**Interpretação**

Representa o montante potencialmente recuperável entre os clientes que o modelo decidiu acionar e que de fato pertencem à classe churn.

### 7.3 Valor Recuperado Esperado

**Definição**

Valor esperado recuperado pela campanha após considerar a taxa de retenção.

**Fórmula**

`Vrec = VR x taxa_de_retenção`

**Interpretação**

Nem todo churner acionado será salvo. Por isso, o valor recuperado esperado corresponde apenas à fração do valor em risco que, em média, a campanha consegue preservar.

### 7.4 Valor Perdido

**Definição**

Valor esperado perdido em churners reais que ficaram fora da campanha por erro do modelo.

**Fórmula**

`VP = Σ (p_i x CLTV_i), para i em FN`

**Interpretação**

Corresponde à oportunidade perdida causada por falsos negativos: clientes que churnariam, mas não foram acionados porque o modelo os classificou como não churn.

### 7.5 Custo Médio por Cliente Acionado

**Definição**

Rateio do custo total da campanha entre todos os clientes efetivamente acionados.

**Fórmula**

`CMCA = custo_total_campanha / (TP + FP)`

**Interpretação**

Essa formulação permite distribuir o investimento total da campanha de forma homogênea entre todos os clientes abordados pelo modelo.

### 7.6 Valor Desperdiçado com Falsos Positivos

**Definição**

Parcela do investimento da campanha consumida com clientes acionados desnecessariamente.

**Fórmula**

`VD = FP x CMCA`

**Interpretação**

Mede o desperdício operacional gerado por falsos positivos: clientes que receberam esforço de retenção, mas que não churnariam de qualquer forma.

### 7.7 Impacto Econômico Líquido

**Definição**

Saldo econômico da estratégia considerando valor recuperado, valor perdido e desperdício com falsos positivos.

**Fórmula**

`IEL = Vrec - VP - VD`

**Interpretação**

Esta métrica mostra se a política de acionamento induzida pelo modelo está protegendo mais valor do que deixando escapar ou desperdiçando.

### 7.8 ROI da Campanha

**Definição**

Retorno sobre o investimento total da campanha, considerando tanto o valor recuperado quanto a perda por falsos negativos e o custo total de acionamento.

**Fórmula**

`ROI = (Vrec - VP - custo_total_campanha) / custo_total_campanha`

**Interpretação**

O ROI responde se, depois de descontar o custo integral da campanha, a estratégia baseada no modelo ainda gera retorno financeiro positivo.

### 7.9 Papel de Cada Célula da Matriz de Confusão

* `TP`: concentram o valor recuperável e alimentam `VR` e `Vrec`.
* `FP`: representam desperdício operacional e alimentam `VD`.
* `FN`: representam oportunidade perdida e alimentam `VP`.
* `TN`: não geram recuperação nem custo direto na formulação atual.

### 7.10 Uso Recomendado no Notebook de Modelo Final

No notebook de consolidação do modelo MVP, recomenda-se:

* comparar os modelos no holdout com base em `Vrec`, `VP`, `VD`, `IEL` e `ROI`;
* analisar o efeito do threshold padrão (`0.5`) versus threshold otimizado;
* testar sensibilidade do resultado econômico a diferentes premissas de taxa de retenção e custo total de campanha.
