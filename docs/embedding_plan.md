# Plano — Trilha separada de `MLP` com embedding para `City`

## Resumo

Implementar uma trilha experimental separada para `MLP` com embedding de `City`, sem tentar encaixar isso no `Pipeline` tabular atual. A ideia é reaproveitar o `FeatureEngineerTransformer`, o `preprocessor` e a lógica de treino da `MLP` já existente, mas separar o fluxo em dois ramos:

- ramo tabular: todas as features exceto geografia bruta e `City`
- ramo categórico: `City` convertida para índices inteiros e enviada para `nn.Embedding`

Essa trilha será usada apenas no notebook de experimentação, como comparação controlada contra a `MLP` tabular atual.

## Implementação

### 1. Nova arquitetura em `src/models/mlp.py`

Adicionar `CityEmbeddingMLP`.

Interface:
- `input_dim`
- `n_cities`
- `embedding_dim`
- `hidden_dim`
- `output_dim=1`

Forward:
1. recebe `x_tabular` e `x_city`
2. aplica `nn.Embedding` em `x_city`
3. concatena embedding com `x_tabular`
4. passa pelo backbone denso
5. retorna logit binário

### 2. Novo wrapper em `src/utils/exp.py`

Adicionar `MLPEmbeddingClassifierWrapper`.

Interface sklearn:
- `fit(X, y, sample_weight=None)`
- `predict_proba(X)`
- `predict(X)`
- `decision_function(X)`

Parâmetros principais:
- `preprocessor`
- `feature_engineer`
- `city_column="City"`
- `geo_drop_columns=("Zip Code", "Latitude", "Longitude", "Lat Long")`
- `embedding_dim=None`
- `unknown_city_index=0`

### 3. Helpers em `src/utils/exp.py`

Adicionar:
- `build_city_vocabulary`
- `encode_city_ids`
- `split_tabular_and_city`
- `make_embedding_dataloader`
- `infer_city_embedding_dim`

### 4. Fluxo de treino

No `fit`:
1. validar `X` como `DataFrame`
2. aplicar `feature_engineer`
3. separar `City`
4. remover geografia bruta do ramo tabular
5. ajustar `preprocessor` só no ramo tabular de treino
6. construir vocabulário só com cidades do treino
7. criar loaders com duas entradas
8. treinar com early stopping

No `predict_proba`:
1. aplicar `feature_engineer.transform`
2. separar `City`
3. transformar tabular com `preprocessor_`
4. mapear cidade para ids com `UNK=0`
5. rodar forward e sigmoid

### 5. Notebook `02_experimentation.ipynb`

Sem reestruturar o notebook.

Na seção `Embedding MLP com City`, comparar:
- `MLP_Tabular`
- `MLP_CityEmbedding`

Usar:
- mesmo `cv`
- mesmo `scoring`
- mesmo formato de tabela
- mesmo bloco de FE estrutural (`round3_fe_params`)

## Testes

### Testes unitários

Cobrir:
- construção do vocabulário com `UNK=0`
- codificação de cidade não vista
- separação entre ramo tabular e ramo `City`
- remoção correta de colunas geográficas do ramo tabular
- shape do forward da `CityEmbeddingMLP`
- `fit/predict_proba` do wrapper em dataset pequeno

### Testes funcionais

Validar no notebook:
- a MLP embedding roda com o mesmo split do baseline
- a tabela final sai no mesmo formato das demais rodadas
- cidades não vistas em folds de validação não quebram o fluxo
- `City` não entra duplicada no ramo tabular

## Assunções

- a trilha de embedding é experimental e separada do pipeline tabular
- `City` continua vindo da base raw no notebook
- o `preprocessor` atual é reaproveitado apenas no ramo tabular
- o wrapper novo encapsula FE + preprocessamento + city indexing para permitir avaliação justa por fold
