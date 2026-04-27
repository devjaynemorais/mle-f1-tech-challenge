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
