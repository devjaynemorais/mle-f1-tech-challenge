# Estratégia de Feature Engineering

## Objetivo

Padronizar a camada de feature engineering da experimentação em torno de `transformers` compatíveis com `scikit-learn`, preservando comparabilidade entre rodadas e reduzindo leakage ao encapsular toda a lógica dentro do `Pipeline`.

## Decisão arquitetural

Adotamos três blocos distintos:

1. `FeatureEngineerTransformer`
2. `GeoTransformer`
3. `MLPEmbedding` como evolução futura para o caminho neural

Essa separação existe porque cada bloco resolve um problema diferente:

- o `FeatureEngineerTransformer` concentra regras tabulares de negócio;
- o `GeoTransformer` concentra apenas estratégias geográficas experimentais;
- o `MLPEmbedding` ficará separado porque embedding não é apenas encoding, mas uma representação aprendida pela rede.

## 1. FeatureEngineerTransformer

### Papel

Centralizar as features tabulares determinísticas que independem de estatísticas do alvo.

### Justificativa

Esse transformer garante que a mesma transformação seja aplicada em treino, validação cruzada e inferência experimental, evitando que a lógica fique espalhada entre notebook, scripts e configuração.

### Features contempladas

- `drop_churn_score`
- `engagement_score`
- `tenure_group_ordinal`
- `tenure_log`
- `contract_ordinal`
- `family_stability`
- `fiber_no_support`
- `support_gap_count`
- `payment_automatic_flag`
- `electronic_check_flag`
- `paperless_echeck_flag`
- `price_pressure_ratio`

### Regras importantes

- `CLTV` não entra no transformer.
  Justificativa: segue como metadata de negócio, não como feature do fluxo-base.

- Geografia não entra no transformer geral.
  Justificativa: `City`, `Zip Code`, `Latitude` e `Longitude` são tratados separadamente no `GeoTransformer`.

- `Tenure_Group` é gerada diretamente em formato ordinal.
  Mapping adotado: `new=2`, `mid=1`, `loyal=0`.

## 2. GeoTransformer

### Papel

Centralizar a camada geográfica da experimentação, controlando tanto o consumo das colunas brutas quanto a geração das features derivadas.

### Colunas tratadas

- `City`
- `Zip Code`
- `Latitude`
- `Longitude`
- `Lat Long` apenas como remoção defensiva quando existir

### Justificativa

O sinal geográfico existe, mas `City` tem alta cardinalidade e muitas categorias raras. Ao concentrar todo o tratamento geográfico em um transformer dedicado, conseguimos:

- testar estratégias concorrentes no mesmo esqueleto de pipeline;
- evitar drops manuais diferentes entre baseline e rounds experimentais;
- reduzir leakage em estratégias supervisionadas;
- remover as colunas geográficas brutas no lugar certo, sem poluir o `preprocessor`.

### Estratégias implementadas

#### `drop`

Remove todas as colunas geográficas brutas.

Usar quando quisermos um baseline sem informação geográfica, mas mantendo o mesmo `X` inicial do notebook.

#### `frequency`

Substitui `City` por sua frequência observada no treino.

Ponto forte:
- simples;
- estável;
- não supervisionado.

Limitação:
- mede prevalência, não risco de churn.

#### `target`

Substitui `City` por uma taxa média de churn suavizada.

Parâmetro principal:
- `target_smoothing`, com default `20`

Ponto forte:
- preserva sinal supervisionado da cidade;
- reduz instabilidade de cidades raras via shrinkage para a média global.

#### `risk_band`

Calcula primeiro uma taxa suavizada de churn por cidade e, só depois, discretiza esse score em `low_risk`, `mid_risk` e `high_risk`.

Ponto forte:
- preserva a ideia de target-based encoding;
- melhora interpretabilidade;
- reduz sensibilidade a categorias raras em comparação com uma média crua.

Observação:
- essa estratégia não implementa IV/WoE;
- ela é uma discretização supervisionada por target com shrinkage por contagem.

#### `zip_region`

Usa `Zip Code` para gerar `Geo_Region` categórica a partir do centróide geográfico aprendido no `fit`.

Funcionamento:
- no `fit`, o transformer aprende o centróide de cada ZIP usando `Latitude` e `Longitude`;
- depois converte esse centróide em uma macro-região (`socal`, `central`, `norcal`);
- no `transform`, aplica apenas o lookup por `Zip Code`;
- ZIPs não vistos ou inválidos vão para `other`.

Ponto forte:
- mais fiel à geografia real do dataset do que uma heurística por faixas de CEP;
- continua interpretável;
- não depende do alvo.

#### `geo_cluster`

Usa `Latitude` e `Longitude` para gerar `Geo_Cluster`.

Funcionamento:
- ajusta `KMeans` no `fit`;
- escolhe `k` automaticamente pelo método do cotovelo;
- transforma o cluster em categoria (`cluster_0`, `cluster_1`, etc.);
- usa `cluster_missing` para coordenadas ausentes.

Ponto forte:
- captura estrutura espacial sem depender do nome da cidade;
- reduz a granularidade geográfica de forma orientada aos dados.

### Regra operacional importante

Independentemente da estratégia, o `GeoTransformer` é o responsável por remover as colunas geográficas brutas depois que a feature derivada foi criada. Isso garante que o `ColumnTransformer` receba apenas a representação final e não as colunas originais.

## 3. Encoders dedicados em `features/encoders.py`

### Papel

Concentrar implementações reutilizáveis e pequenas, deixando o `GeoTransformer` como orquestrador de estratégia.

### Encoders disponíveis

- `FrequencyEncoder`
- `TargetEncoder`
- `RiskBandEncoder`
- `ZipRegionEncoder`
- `GeoClusterEncoder`

### Justificativa

Separar encoder de transformer melhora manutenção, testes e reuso:

- o `GeoTransformer` decide a estratégia;
- cada encoder implementa a lógica estatística ou geográfica específica;
- futuras evoluções podem reaproveitar esses blocos sem duplicação.

## 4. Relação com o notebook e com produção

### Notebook de experimentação

O notebook mantém a mesma estrutura geral. A mudança é apenas pontual nas linhas que instanciam o `GeoTransformer`, trocando estratégias e parâmetros quando necessário.

### Pipeline de produção

Essas estratégias geográficas novas pertencem apenas ao fluxo de experimentação neste momento.

Motivo:
- não alteramos `make_dataset.py`;
- o pipeline produtivo atual continua sem `Zip Code`, `Latitude` e `Longitude`;
- a base raw do notebook continua sendo o ambiente correto para testar `zip_region` e `geo_cluster`.

## 5. Caminho futuro: MLPEmbedding

Embedding de cidade continuará fora do `GeoTransformer`.

Fluxo esperado no futuro:

1. indexar a cidade em IDs inteiros;
2. passar esses IDs para uma camada `nn.Embedding`;
3. concatenar o embedding com as demais features da MLP.

Essa decisão foi mantida porque o embedding depende do loop de treino da rede e não deve ser confundido com os encodings tabulares clássicos.

## Benefícios esperados

- Menos acoplamento entre notebook e lógica de transformação.
- Maior consistência entre treino, validação e inferência experimental.
- Menor risco de leakage em estratégias supervisionadas.
- Melhor comparabilidade entre baseline, FE estrutural e rounds geográficos.
- Base mais limpa para evoluir depois para logs de experimento, seleção de features e embeddings específicos da MLP.

