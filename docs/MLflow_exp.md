# Plano Revisado de Tracking de Experimentos no MLflow para o `02_experimentation.ipynb`

## Resumo

Vamos implementar uma primeira versão do tracking no MLflow que seja:

- **enxuta e navegável**
- **fiel ao notebook atual**
- **sem refatoração pesada**
- **segura para o `.ipynb`**
- **100% compatível com UTF-8**

A implementação vai cobrir:

- baseline
- Round 1 e Round 2 de feature engineering
- Round 3 de encoding geográfico
- Round 4 de feature selection
- tuning etapa 1 com `RandomizedSearchCV` para `MLP` e `XGBoost`
- tuning etapa 2 com `Optuna` para `MLP` e `XGBoost`

Também vamos:

- logar índices dos splits e metadados sincronizados desde o baseline
- criar helpers fora do notebook para não sobrecarregar o `.ipynb`
- implementar o `Optuna`, que hoje ainda está ausente
- incluir o **grid paramétrico reduzido** como artefato tanto na etapa 1 quanto na etapa 2 do tuning
- usar **parada por timeout + convergência** no `Optuna`
- tratar **UTF-8 como hard constraint** da implementação

## Constraint obrigatória: UTF-8

Essa v1 terá a seguinte restrição formal:

- todo arquivo novo será salvo em **UTF-8**
- toda alteração no notebook [02_experimentation.ipynb](/abs/path/c:/Users/jooar/ds/portfolio/mle-f1-tech-challenge-fiap/notebooks/02_experimentation.ipynb) deve **preservar UTF-8**
- a lógica nova será concentrada em helpers fora do notebook para minimizar regravação do `.ipynb`
- vamos evitar alterar texto do notebook além do estritamente necessário
- a verificação final incluirá:
  - leitura do notebook com `nbformat` sem erro
  - smoke check do notebook ainda parseando normalmente
  - inspeção para garantir ausência de caracteres estranhos ou corrupção visual

## Arquitetura e helpers

### Novo módulo `src/utils/mlflow_tracking.py`
Responsável por encapsular o tracking do notebook.

Funções:
- `set_tracking(tracking_uri: str) -> None`
- `ensure_experiment(experiment_name: str) -> str`
- `start_experiment_run(experiment_name: str, run_name: str, nested: bool = False, tags: dict | None = None)`
- `log_params_safe(params: dict) -> None`
- `log_metrics_safe(metrics: dict[str, float | int]) -> None`
- `log_dataframe_artifact(df: pd.DataFrame, artifact_name: str) -> None`
- `log_json_artifact(payload: dict | list, artifact_name: str) -> None`
- `log_split_artifacts(train_val_idx, test_idx, metadata_train_val, metadata_test) -> None`
- `best_row_to_metrics(best_row: pd.Series | dict, prefix: str = "best") -> dict`

### Novo módulo `src/utils/optuna_search.py`
Responsável por tirar do notebook toda a lógica do `Optuna`.

Funções:
- `suggest_params_from_space(trial, search_space: dict) -> dict`
- `build_mlp_optuna_pipeline(preprocessor, fe_params: dict, params: dict) -> Pipeline`
- `build_xgb_optuna_pipeline(preprocessor, fe_params: dict, params: dict, y_reference) -> Pipeline`
- `evaluate_candidate_cv(estimator, X, y, cv, scoring) -> dict`
- `make_optuna_objective(model_name: str, pipeline_factory, X, y, cv, scoring, search_space: dict, trial_records: list)`
- `build_trials_df(trial_records: list[dict]) -> pd.DataFrame`
- `build_convergence_history(trials_df: pd.DataFrame, objective_col: str = "pr_auc_mean") -> pd.DataFrame`
- `should_stop_for_convergence(trials_df, patience_trials: int, min_improvement: float) -> bool`
- `run_optuna_study(...) -> tuple[study, trials_df, convergence_df, best_params, best_result]`

### Princípio de implementação
- o notebook continua sendo o orquestrador
- os cálculos existentes permanecem onde já estão
- o tracking entra como células curtas ao final de cada etapa
- não vamos criar artefatos artificiais além do necessário
- qualquer tabela hoje apenas exibida e que precise ir para o MLflow será materializada em variável antes do log

## Estrutura completa dos experimentos

### 1. Experimento `tc-f1-nb02-baselines`

**Run pai**
- `baseline_inicial`

**Nested runs**
- `Dummy`
- `LogisticRegression`
- `DecisionTree`
- `RandomForest`
- `XGBoost`
- `MLP`

**Run pai `baseline_inicial`**
- Params:
  - `phase=baseline_inicial`
  - `cv_folds`
  - `primary_metric=pr_auc`
  - `random_state=42`
- Artefatos:
  - `results_cv.csv`
  - `mlp_wrapper_validation.csv`
  - `train_val_indices.csv`
  - `test_indices.csv`
  - `metadata_train_val.csv`
  - `metadata_test.csv`

**Nested runs**
- Params:
  - `model_name`
  - hiperparâmetros do modelo
  - flags de FE do baseline
- Metrics:
  - `pr_auc_mean`
  - `pr_auc_std`
  - `roc_auc_mean`
  - `recall_mean`
  - `precision_mean`
  - `f1_mean`
  - `fit_time_mean_s`
  - `score_time_mean_s`

---

### 2. Experimento `tc-f1-nb02-feature-engineering`

**Runs pai**
- `round1_feature_engineering`
- `round2_feature_engineering`

**Nested runs**
- `LogisticRegression`
- `XGBoost`
- `MLP`

**Run pai**
- Params:
  - `phase=feature_engineering`
  - `round=1` ou `round=2`
  - flags de FE do round
  - `uses_churn_score=False` no Round 1
  - `uses_churn_score=True` no Round 2
- Artefatos:
  - `results_cv_round1.csv`
  - `results_cv_round2.csv`

**Nested runs**
- Params:
  - `model_name`
  - flags de FE usadas
- Metrics:
  - `pr_auc_mean`
  - `pr_auc_std`
  - `roc_auc_mean`
  - `recall_mean`
  - `precision_mean`
  - `f1_mean`
  - `fit_time_mean_s`
  - `score_time_mean_s`

---

### 3. Experimento `tc-f1-nb02-round3-city-encoding`

**Run de resumo**
- `round3_summary`

**Runs pai**
- `round3_logistic_regression`
- `round3_xgboost`
- `round3_mlp`

**Nested runs por estratégia**
- LogisticRegression:
  - `frequency`
  - `target`
  - `geo_cluster`
  - `zip_region`
  - `risk_band`
- XGBoost:
  - `frequency`
  - `target`
  - `geo_cluster`
  - `zip_region`
  - `risk_band`
- MLP:
  - `frequency`
  - `target`
  - `geo_cluster`
  - `zip_region`
  - `risk_band`
  - `city_embedding`

**Run `round3_summary`**
- Artefatos:
  - `round3_results_all.csv`

**Run pai do modelo**
- Params:
  - `phase=round3_city_encoding`
  - `model_name`
- Metrics:
  - `best_pr_auc_mean`
  - `best_roc_auc_mean`
  - `best_recall_mean`
- Artefatos:
  - `round3_results_<model>.csv`

**Nested run da estratégia**
- Params:
  - `model_name`
  - `strategy_name`
  - parâmetros específicos quando existirem
- Metrics:
  - `pr_auc_mean`
  - `roc_auc_mean`
  - `recall_mean`
  - `precision_mean`
  - `f1_mean`
  - `fit_time_mean_s`
  - `score_time_mean_s`

---

### 4. Experimento `tc-f1-nb02-feature-selection`

**Run de resumo**
- `selectkbest_summary`

**Runs**
- `selectkbest_logistic_regression`
- `selectkbest_xgboost`
- `selectkbest_mlp`
- `l1_based_logistic_regression`

**Run `selectkbest_summary`**
- Artefatos:
  - `results_fs.csv`
  - `round4_k_candidates.json`

**Runs `selectkbest_*`**
- Params:
  - `phase=feature_selection`
  - `method=selectkbest`
  - `model_name`
  - `k_grid`
  - `selectors_tested`
  - `sort_order=pr_auc>roc_auc>recall`
- Metrics:
  - `best_pr_auc_mean`
  - `best_pr_auc_std`
  - `best_roc_auc_mean`
  - `best_recall_mean`
  - `best_precision_mean`
  - `best_f1_mean`
  - `best_fit_time_mean_s`
  - `best_score_time_mean_s`
  - `best_k`
  - `best_selector`
- Artefatos:
  - `results_fs_<model>.csv`
  - `selector_summary_<model>.csv`
  - `k_summary_<model>.csv`
  - `k_best_<model>.csv`

**Run `l1_based_logistic_regression`**
- Params:
  - `phase=feature_selection`
  - `method=l1_based_selection`
  - `model_name=LogisticRegression`
  - `min_features_threshold=10`
  - `c_grid`
- Metrics:
  - `best_pr_auc_mean`
  - `best_roc_auc_mean`
  - `best_recall_mean`
  - `best_precision_mean`
  - `best_f1_mean`
  - `best_fit_time_mean_s`
  - `best_score_time_mean_s`
  - `best_c`
  - `best_n_selected_features`
- Artefatos:
  - `results_l1_logistic_regression.csv`
  - `l1_describe_logistic_regression.csv`

---

### 5. Experimento `tc-f1-nb02-tuning-stage1-randomsearch`

**Runs**
- `randomsearch_mlp`
- `randomsearch_xgboost`

**Run `randomsearch_mlp`**
- Params:
  - `phase=tuning_stage1`
  - `search_type=randomized_search`
  - `model_name=MLP`
  - `n_iter=100`
  - `cv_folds`
  - `refit_metric=pr_auc`
  - `random_state=42`
  - `uses_selectkbest=True`
  - `fixed_max_epochs=80`
  - `fixed_patience=16`
  - `fixed_threshold=0.5`
- Metrics:
  - `best_pr_auc_mean`
  - `best_pr_auc_std`
  - `best_roc_auc_mean`
  - `best_recall_mean`
  - `best_precision_mean`
  - `best_f1_mean`
  - `best_fit_time_mean_s`
  - `best_score_time_mean_s`
- Params adicionais da best:
  - `best_k`
  - `best_activation`
  - `best_hidden_dim`
  - `best_dropout`
  - `best_lr`
  - `best_weight_decay`
  - `best_batch_size`
- Artefatos:
  - `results_mlp_random_search.csv`
  - `best_mlp_result.json`
  - `best_mlp_params.json`
  - `best_mlp_selected_features.json`
  - `mlp_top20.csv`
  - `mlp_discrete_lift_summary.csv`
  - `mlp_numeric_ranges_summary.csv`
  - `mlp_param_effect_summary.csv`
  - `mlp_interactions_summary.csv`
  - `mlp_optuna_search_space.json`

**Run `randomsearch_xgboost`**
- Params:
  - `phase=tuning_stage1`
  - `search_type=randomized_search`
  - `model_name=XGBoost`
  - `n_iter=100`
  - `cv_folds`
  - `refit_metric=pr_auc`
  - `random_state=42`
  - `uses_selectkbest=False`
  - `feature_selection_mode=intrinsic`
- Metrics:
  - `best_pr_auc_mean`
  - `best_pr_auc_std`
  - `best_roc_auc_mean`
  - `best_recall_mean`
  - `best_precision_mean`
  - `best_f1_mean`
  - `best_fit_time_mean_s`
  - `best_score_time_mean_s`
- Params adicionais da best:
  - melhores hiperparâmetros da run
- Artefatos:
  - `results_xgb_random_search.csv`
  - `best_xgb_result.json`
  - `best_xgb_params.json`
  - `xgb_top20.csv`
  - `xgb_discrete_lift_summary.csv`
  - `xgb_param_effect_summary.csv`
  - `xgb_optuna_search_space.json`

---

### 6. Experimento `tc-f1-nb02-tuning-stage2-optuna`

**Runs**
- `optuna_mlp`
- `optuna_xgboost`

**Estratégia**
- sem nested runs por trial
- uma run por estudo/modelo
- timeout + convergência
- trials completos e histórico de convergência como artefatos
- grid reduzido do Optuna logado também nesta etapa

**Regra de parada**
- `timeout_seconds=3600`
- `convergence_patience_trials=50`
- `convergence_min_improvement=5e-3`

O estudo para quando:
- atingir 1 hora; ou
- completar 50 trials consecutivos sem melhorar o melhor `pr_auc_mean` em pelo menos `0.005`

Esse critério replica, no nível de trials, a lógica de `min_delta` usada na MLP.

**Run `optuna_mlp`**
- Params:
  - `phase=tuning_stage2`
  - `search_type=optuna`
  - `model_name=MLP`
  - `study_direction=maximize`
  - `objective_metric=pr_auc`
  - `timeout_seconds=3600`
  - `convergence_patience_trials=50`
  - `convergence_min_improvement=5e-3`
  - `sampler_name=TPESampler`
  - `random_state=42`
- Metrics:
  - `best_pr_auc_mean`
  - `best_roc_auc_mean`
  - `best_recall_mean`
  - `best_precision_mean`
  - `best_f1_mean`
  - `best_fit_time_mean_s`
  - `best_score_time_mean_s`
  - `n_trials_completed`
  - `best_trial_number`
  - `total_runtime_s`
- Artefatos:
  - `optuna_trials_mlp.csv`
  - `optuna_top_trials_mlp.csv`
  - `optuna_convergence_history_mlp.csv`
  - `best_mlp_optuna_params.json`
  - `best_mlp_optuna_result.json`
  - `mlp_optuna_search_space.json`

**Run `optuna_xgboost`**
- mesmo padrão do `MLP`
- Artefatos:
  - `optuna_trials_xgb.csv`
  - `optuna_top_trials_xgb.csv`
  - `optuna_convergence_history_xgb.csv`
  - `best_xgb_optuna_params.json`
  - `best_xgb_optuna_result.json`
  - `xgb_optuna_search_space.json`

## Arquitetura detalhada do Optuna

### Código que fica no notebook
Só ficará no notebook:
- `mlp_optuna_search_space`
- `xgb_optuna_search_space`
- chamadas do helper do `Optuna`
- chamadas do helper de MLflow
- `display(...)` dos resultados finais

Toda a lógica da busca ficará em `src/utils/optuna_search.py`.

### Fluxo do Optuna para MLP
1. Ler `mlp_optuna_search_space`
2. Montar pipeline com:
   - `FeatureEngineerTransformer(**round4_fe_params)`
   - `GeoTransformer(strategy="drop")`
   - `preprocessor`
   - `SelectKBest`
   - `StandardScaler(with_mean=False)`
   - `MLPClassifierWrapper`
3. Em cada trial:
   - sugerir parâmetros
   - reconstruir pipeline
   - rodar `cross_validate`
   - calcular métricas médias e desvios
   - salvar resultado em `trial_records`
   - atualizar melhor valor observado
   - verificar convergência com `5e-3`
   - retornar `pr_auc_mean`
4. Ao final:
   - montar `trials_df`
   - montar `top_trials_df`
   - montar `convergence_df`
   - obter `best_params`
   - obter `best_result`
   - logar tudo no MLflow

### Fluxo do Optuna para XGBoost
Mesmo padrão, mas sem `SelectKBest` e sem `StandardScaler`. O pipeline usa:
- `FeatureEngineerTransformer(**round4_fe_params)`
- `GeoTransformer(strategy="drop")`
- `preprocessor`
- `XGBClassifier`

### Convenção do search space
- listas:
  - `trial.suggest_categorical`
- dicionário contínuo com `low/high/log`:
  - `trial.suggest_float`
- dicionário inteiro com `low/high/step`:
  - `trial.suggest_int`

### Sampler e pruning
- `optuna.samplers.TPESampler(seed=42)`
- sem pruner na v1
- parada controlada por timeout + convergência

## O que falta implementar

### No notebook
- inserir chamadas de MLflow em todas as etapas prontas
- materializar em variáveis as tabelas analíticas hoje só exibidas:
  - `mlp_discrete_lift_summary`
  - `mlp_numeric_ranges_summary`
  - `mlp_param_effect_summary`
  - `mlp_interactions_summary`
  - `xgb_discrete_lift_summary`
  - `xgb_param_effect_summary`
- persistir `results_cv` com nomes específicos por round durante o logging
- persistir `describe()` do L1 em dataframe
- preencher as células vazias do `Optuna`

### Fora do notebook
- criar `src/utils/mlflow_tracking.py`
- criar `src/utils/optuna_search.py`

### Na etapa de Optuna
- implementar `Optuna` para `MLP`
- implementar `Optuna` para `XGBoost`
- gerar histórico de convergência
- subir grid reduzido na etapa 2
- logar runs dos estudos no MLflow

## Testes e validação

### Testes
- `tests/test_mlflow_tracking.py`
  - log de dataframe
  - log de JSON
  - log de split indices + metadata
  - extração da best row

- `tests/test_optuna_search.py`
  - interpretação de search space
  - geração de `trials_df`
  - geração de `convergence_df`
  - parada por convergência com `5e-3`

### Smoke checks
- notebook continua parseando
- células do `Optuna` deixam de estar vazias
- nomes dos experimentos, runs e artefatos batem com o plano
- notebook continua íntegro em UTF-8
- leitura via `nbformat` sem erro
- ausência de caracteres estranhos após a edição

## Assumptions e defaults

- o notebook continua como orquestrador principal
- o tracking da v1 não serializa o modelo final treinado
- a serialização do modelo refitado após calibração, threshold e análise econômica fica para etapa posterior
- não vamos gerar gráficos ou relatórios extras só para o MLflow
- o `Round 3` fica em experimento separado
- o `Optuna` usa uma run por estudo/modelo, sem nested por trial
- o `Optuna` para por `timeout + convergência`
- o padrão de convergência é:
  - `50` trials de patience
  - `5e-3` de melhoria mínima
- **UTF-8 é requisito obrigatório de implementação para notebook e helpers**
