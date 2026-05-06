# Model Card - Churn Prediction MLP Optuna

## Descricao do Modelo

**Nome:** MLP Optuna para Predicao de Churn  
**Versao:** 1.1.0  
**Tipo:** Classificacao binaria (`churn = 1` / `nao churn = 0`)  
**Framework:** PyTorch + Scikit-Learn + MLflow  
**Status:** Candidato a producao serializado no MLflow  
**Model URI:** `runs:/71b3084f4c444df4a2470992a83cbe92/model`

O modelo final foi consolidado no notebook [03_modelo_mvp.ipynb](/C:/Users/jooar/ds/portfolio/mle-f1-tech-challenge-fiap/notebooks/03_modelo_mvp.ipynb), com serializacao registrada no experimento `tc-f1-nb03-MLP-production-performance`.

---

## Uso Pretendido

**Objetivo primario:** identificar clientes de telecom com risco elevado de cancelamento para priorizacao de campanhas de retencao.

**Usuarios previstos:**
- Times de CRM
- Marketing
- Sucesso do Cliente
- Operacao de retencao

**Casos fora do escopo:**
- Predicao de churn em outros setores sem retreinamento
- Decisoes automatizadas de alto impacto sem supervisao humana
- Uso como unico criterio para negar servico, credito ou beneficios

---

## Dados e Features

**Dataset:** Telco Customer Churn - IBM (`data/raw/Telco_customer_churn.xlsx`)  
**Base tratada:** `data/interim/telecom_clean.csv`  
**Tamanho total:** 7.043 registros  
**Classe positiva:** aproximadamente 26% de churners  
**Holdout:** split fixado e reconstruido a partir dos artefatos logados no MLflow

O pipeline final usa:
- engenharia de atributos da rodada final (`round 4`)
- `GeoTransformer(strategy="drop")`
- `ColumnTransformer` com `OneHotEncoder` para categoricas
- `SelectKBest(f_classif)` para selecao supervisionada de atributos
- `StandardScaler(with_mean=False)`
- wrapper `MLPClassifierWrapper`

**Quantidade final de features selecionadas:** `44`

Entre os atributos finais selecionados, destacam-se:
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Contract`
- `Tenure Months`
- `Monthly Charges`
- `Total Charges`
- `service_score`
- `Support_Gap_Count`
- `Price_Pressure_Ratio`

O atributo `Gender` nao foi mantido no pipeline final. Durante a EDA e a selecao de variaveis, ele mostrou baixa variabilidade explicativa e baixo poder preditivo relativo para churn.

---

## Arquitetura e Hiperparametros

### Modelo final selecionado pelo Optuna

- `selector__k = 44`
- `model__activation = tanh`
- `model__hidden_dim = 54`
- `model__dropout = 0.3`
- `model__lr = 0.000718278415521642`
- `model__weight_decay = 0.0`
- `model__batch_size = 128`

### Parametros fixos do wrapper no pipeline final

- `max_epochs = 80`
- `patience = 16`
- `min_delta = 1e-3`
- `val_size = 0.15`
- `random_state = 42`
- `threshold de producao = 0.35`

### Criterio economico adotado no notebook 03

- `activation_cost = BRL 50`
- `retention_rate = 0.10`

O threshold de producao nao permaneceu em `0.5`. Ele foi explicitamente redefinido para `0.35` na etapa final de serializacao para refletir a politica de decisao escolhida para rollout.

---

## Performance do Modelo

### Desenvolvimento (cross-validation / OOF)

Melhor configuracao da `MLP Optuna` no processo de tuning:

| Metrica | Valor |
|---|---:|
| PR-AUC media | 0.9401 |
| ROC-AUC media | 0.9763 |
| Recall medio | 0.9273 |
| Precision media | 0.7803 |
| F1 medio | 0.8469 |
| Fit time medio (s) | 2.6048 |
| Score time medio (s) | 0.0285 |

### Validacao final e producao

O candidato de producao foi:
- refitado em todo o `train_val`
- avaliado em `X_test`
- serializado no MLflow com `threshold = 0.35`

As metricas finais de holdout e os artefatos associados foram registradas no MLflow na run de serializacao do modelo e nas runs de `holdout_evaluation`, `shap` e `fairness` do experimento `tc-f1-nb03-MLP-production-performance`.

---

## Logica Economica

Durante a consolidacao do notebook 03, a avaliacao economica foi reformulada para usar **custo por acionamento** em vez de custo total fixo de campanha.

Premissas utilizadas:
- `activation_cost = BRL 50`
- `retention_rate = 10%`

Principais componentes:
- `VR`: valor em risco
- `Vrec`: valor recuperado
- `VP`: perda por omissao
- `VD`: desperdicio com falsos positivos
- `IEL`: impacto economico liquido
- `ROI`: retorno sobre o custo total de acionamento

Essa mudanca foi importante para evitar que thresholds artificialmente muito baixos parecessem otimos apenas por diluicao de custo fixo.

---

## Ablacao Economica

Foi testada uma ablacao economica com ponderacao do treino por `CLTV` via `sample_weight`, com o objetivo de aproximar o objetivo estatistico do objetivo financeiro da campanha.

Cenarios comparados:
- `p` puro, sem ponderacao por `CLTV`
- `p` ponderado por `CLTV` no treino

Para a `MLP Optuna`, o cenario ponderado nao trouxe ganho suficiente no contexto testado e reduziu os indicadores economicos principais em relacao ao cenario puro.

No cenario sem ponderacao, a `MLP Optuna` apresentou:
- `IEL = 329343.52`
- `ROI = 2.52`

Conclusao metodologica:

> A ponderacao do treino por `CLTV` foi testada como aproximacao entre objetivo estatistico e objetivo economico, mas nao trouxe ganho para a `MLP Optuna`, elevando o custo de omissao (`VP`) e reduzindo o retorno final da estrategia.

Por isso, o candidato ao MVP foi mantido com treinamento sem ponderacao por `CLTV`, e o teste ponderado passou a ser tratado como ablacao economica documentada.

---

## Fairness, Limitacoes e Vieses Conhecidos

### Escopo da analise

A analise de fairness foi executada no `holdout`, com comparacao por grupos em:
- `Senior Citizen`
- `Partner`
- `Dependents`
- `Contract`

Tambem foi feita comparacao auxiliar com `threshold = 0.5`, e os desalinhamentos observados permaneceram de forma semelhante. Isso sugere que os gaps nao foram criados apenas pela escolha do threshold final.

### Principais achados

#### 1. Gender

- `Gender` nao foi selecionado entre as features finais.
- Na EDA e na analise de IV, a variavel mostrou baixa relevancia preditiva.
- Isso reduz dependencia explicita de um atributo sensivel no pipeline final.

**Importante:** a ausencia de `Gender` no modelo nao garante ausencia de vies. O modelo ainda pode reproduzir efeitos indiretos por proxies ou por combinacoes de outras variaveis.

#### 2. Senior Citizen

No threshold final analisado no holdout, os gaps por `Senior Citizen` foram moderados:
- `demographic_parity_difference = 0.2034`
- `equalized_odds_difference = 0.0520`
- `equalized_odds_ratio = 0.7061`

Leitura:
- houve diferenca de exposicao a campanha entre grupos
- o gap de `recall` foi relativamente pequeno
- nao surgiu evidencia forte o suficiente para exigir mitigacao imediata

**Tratamento recomendado:** monitoramento recorrente em producao.

#### 3. Partner

Resultados observados:
- `demographic_parity_difference = 0.2059`
- `equalized_odds_difference = 0.1131`
- `equalized_odds_ratio = 0.4100`

Leitura:
- ha diferenca relevante de `selection_rate` e `FPR`
- o `F1` por grupo permaneceu parecido

**Interpretacao recomendada:** alerta secundario de desempenho por subgrupo, mais proximo de segmentacao operacional do que de fairness regulatoria prioritaria.

#### 4. Dependents

Esse foi o ponto mais sensivel da analise:
- `demographic_parity_difference = 0.3296`
- `equalized_odds_difference = 0.2127`
- `equalized_odds_ratio = 0.2415`

Leitura:
- o grupo `Dependents = Yes` apresentou recall inferior
- a taxa de falsos negativos foi materialmente maior nesse grupo
- isso implica maior risco de deixar escapar churners desse segmento

**Interpretacao:** este e o principal vies operacional observado no modelo atual.  
**Decisao para o MVP:** documentar como limitacao conhecida e abrir investigacao futura, sem mitigacao imediata nesta entrega.

#### 5. Contract

Resultados observados:
- `demographic_parity_difference = 0.5221`
- `equalized_odds_difference = 0.3051`
- `equalized_odds_ratio = 0.0732`

Leitura:
- os gaps sao muito altos entre `Month-to-month`, `One year` e `Two year`
- o desempenho e bastante heterogeneo por regime contratual

**Interpretacao correta:** isso deve ser tratado principalmente como heterogeneidade de performance por segmento de negocio, nao como fairness sensivel classica.

### Limitacoes de fairness

1. **Nao houve mitigacao fairness-aware nesta entrega.**  
   O modelo foi levado para serializacao e rollout com documentacao dos gaps, mas sem ajuste por threshold por grupo, reranking ou tecnicas fairness-aware.

2. **A principal limitacao operacional esta em `Dependents`.**  
   O grupo `Yes` ficou mais exposto a falsos negativos no holdout.

3. **`Contract` mostrou forte heterogeneidade de performance.**  
   Mesmo nao sendo atributo sensivel classico, isso pode levar a respostas desbalanceadas da campanha por perfil contratual.

4. **`Senior Citizen` exige monitoramento continuo.**  
   O gap observado nao justificou mitigacao imediata, mas tambem nao deve ser ignorado.

5. **O threshold final foi otimizado para criterio economico.**  
   Isso alinha melhor a politica de campanha ao negocio, mas pode deslocar trade-offs de selecao entre grupos.

### Recomendacao futura

As proximas iteracoes podem investigar:
- fairness-aware post-processing
- reranking por valor esperado com restricoes de fairness
- thresholds por grupo apenas em ambiente experimental
- revisao de proxies e engenharia de atributos
- metricas economicas por subgrupo em producao

---

## Cenarios de Falha

| Cenario | Efeito | Mitigacao recomendada |
|---|---|---|
| Dados de entrada com schema incorreto | erro de predicao ou inferencia inconsistente | validar schema antes da inferencia |
| Drift de distribuicao | queda de recall, precision e ROI | monitorar PSI e retreinar quando necessario |
| CLTV fora da distribuicao esperada | degradacao da leitura economica | revisar limites e validacao do CLTV em producao |
| Heterogeneidade por grupo | comportamento desigual por segmento | monitorar fairness e metricas por subgrupo |
| Desserializacao em ambiente nao confiavel | risco de execucao arbitraria com `cloudpickle` | carregar o modelo apenas em ambiente controlado; avaliar `skops` futuramente |

---

## Plano de Monitoramento

### Metricas recomendadas

| Metrica | Frequencia | Acao sugerida |
|---|---|---|
| PR-AUC / ROC-AUC | semanal ou quinzenal | investigar degradacao persistente |
| Recall | semanal | revisar threshold ou drift se houver queda material |
| ROI / IEL | por campanha | recalibrar politica de acionamento se houver perda de retorno |
| Selection rate por grupo | mensal | comparar com baseline de rollout |
| FNR por `Dependents` | mensal | prioridade alta de monitoramento |
| Equalized odds por `Senior Citizen` | mensal | manter acompanhamento |
| PSI das principais features | mensal | acionar retreinamento se drift for relevante |

### Playbook resumido

1. Validar drift de entrada.
2. Comparar metricas tecnicas e economicas com a run baseline de producao.
3. Revisar fairness por `Dependents`, `Senior Citizen` e `Contract`.
4. Reavaliar threshold se o padrao de churn ou o custo de acionamento mudar.

---

## Informacoes Tecnicas

- **Repositorio:** POS MLE - Tech Challenge Fase 1
- **Notebook de consolidacao:** [03_modelo_mvp.ipynb](/C:/Users/jooar/ds/portfolio/mle-f1-tech-challenge-fiap/notebooks/03_modelo_mvp.ipynb)
- **Run upstream do Optuna MLP:** `3132d9b960c947afa75163c1ca0679b8`
- **Run de serializacao para producao:** `71b3084f4c444df4a2470992a83cbe92`
- **Formato de serializacao:** `mlflow.sklearn` com `cloudpickle`
- **Threshold de producao:** `0.35`
- **Ultima atualizacao:** `2026-05-02`
- **Contato:** `devmoraislacerda@gmail.com`
