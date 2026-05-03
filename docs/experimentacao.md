# Metodologia de Experimentação para Modelos de Churn

## Objetivo

Estabelecer um fluxo estruturado e reprodutível para desenvolvimento de modelos de churn, equilibrando performance técnica, risco de leakage e viabilidade operacional.

## Princípios norteadores

1. Começar simples e evoluir com evidência.
2. Comparar modelos nas mesmas condições experimentais.
3. Separar predição de decisão de negócio.
4. Tratar geografia como bloco experimental próprio.
5. Preservar a mesma espinha dorsal de pipeline entre baseline e rounds avançados.

## Protocolo de validação

**Validação cruzada estratificada com `StratifiedKFold`**

Motivo:
- a base é moderadamente desbalanceada;
- queremos medir ganho de feature engineering sem misturar técnicas de balanceamento neste momento.

## Visão geral do workflow

### Fase 1 — Baseline inicial

Objetivo:
- medir o sinal preditivo inicial da base;
- comparar `Dummy`, `LogisticRegression`, `MLP` e modelos de árvore;
- escolher o benchmark tabular.

Saída esperada:
- tabela comparativa com `PR-AUC`, `ROC-AUC`, `Recall`, `Precision`, `F1`, tempos e estabilidade de CV.

### Fase 2 — Feature engineering estrutural

Objetivo:
- adicionar features de negócio derivadas da EDA sem misturar score legado nem tratamento geográfico especial.

Features estruturais:
- `tenure_group_ordinal`
- `tenure_log`
- `service_score`
- `contract_ordinal`
- `family_stability`
- `fiber_no_support`
- `support_gap_count`
- `payment_automatic_flag`
- `electronic_check_flag`
- `paperless_echeck_flag`
- `price_pressure_ratio`

Regras:
- `Churn Score` fica fora nesta fase;
- geografia também fica fora desta fase principal.

### Fase 3 — Rodadas geográficas controladas

Objetivo:
- testar o sinal geográfico separadamente via `GeoTransformer`, sem reescrever o notebook nem alterar o pipeline produtivo.

Estratégias disponíveis:
- `drop`
- `frequency`
- `target`
- `risk_band`
- `zip_region`
- `geo_cluster`

Por que essa fase é separada:
- `target` e `risk_band` dependem do alvo;
- `zip_region` e `geo_cluster` dependem de colunas geográficas brutas;
- queremos preservar comparabilidade entre rounds.

### Fase 4 — Ablação com `Churn Score`

Objetivo:
- medir separadamente o efeito de incluir `Churn Score`, já que ele representa informação derivada de outro modelo.

### Fase 5 — Seleção de features

Objetivo:
- otimizar subconjuntos de features mantendo toda a lógica dentro do `Pipeline`.

Estratégias:
- `SelectKBest` com `f_classif`
- `SelectKBest` com `mutual_info_classif`

Motivo:
- evita leakage;
- mantém a seleção dentro de cada fold;
- facilita comparação entre `LogisticRegression`, `MLP` e benchmark de árvore.

### Fase 6 — Tunagem

Objetivo:
- aprofundar principalmente a `MLP` após estabilizar a melhor versão técnica do dataset.

### Fase 7 — Avaliação econômica com `CLTV`

Objetivo:
- usar `CLTV` como metadata de negócio para avaliar retorno financeiro da priorização.

Regra:
- `CLTV` não entra como feature do fluxo-base;
- ele entra depois como critério econômico de comparação.

## Camada geográfica experimental

### Como funciona

Toda a geografia fica centralizada no `GeoTransformer`, que controla:
- consumo de `City`, `Zip Code`, `Latitude` e `Longitude`;
- geração da feature derivada;
- remoção das colunas geográficas brutas antes do `preprocessor`.

### Pontos fortes da abordagem

- reduz leakage em estratégias supervisionadas;
- evita drops manuais diferentes por rodada;
- mantém baseline e experimentos no mesmo esqueleto de pipeline;
- facilita testes controlados com mudanças pontuais no notebook.

### Estratégias

#### `drop`
- remove toda a geografia bruta;
- serve como baseline comparável.

#### `frequency`
- usa a frequência observada da cidade;
- simples e estável;
- não supervisionado.

#### `target`
- usa taxa média de churn por cidade com smoothing;
- default de smoothing: `20`;
- reduz instabilidade de cidades raras.

#### `risk_band`
- calcula primeiro uma taxa suavizada de churn por cidade;
- depois discretiza em `low_risk`, `mid_risk` e `high_risk`;
- melhora interpretabilidade.

Observação:
- `risk_band` não implementa IV/WoE;
- ele é um target-based encoding discretizado com shrinkage por contagem.

#### `zip_region`
- transforma `Zip Code` em `Geo_Region`;
- aprende o centróide geográfico de cada ZIP a partir de `Latitude` e `Longitude` no treino experimental;
- é interpretável e não supervisionado.

#### `geo_cluster`
- usa `Latitude` e `Longitude`;
- ajusta `KMeans` dentro do transformer;
- escolhe `k` automaticamente pelo método do cotovelo;
- gera `Geo_Cluster`.

## Relação com o notebook

O notebook de experimentação mantém a mesma estrutura geral. Os ajustes esperados são apenas pontuais:
- trocar nomes de estratégia;
- atualizar parâmetros do `GeoTransformer`;
- manter a leitura da base raw para suportar `Zip Code`, `Latitude` e `Longitude`.

Não é necessário reorganizar seções nem criar um novo fluxo paralelo.

## Relação com produção

Essas estratégias geográficas novas pertencem apenas à camada de experimentação neste momento.

Motivo:
- não alteramos `make_dataset.py`;
- o pipeline produtivo atual continua sem `Zip Code`, `Latitude` e `Longitude`;
- isso preserva estabilidade operacional enquanto a hipótese geográfica ainda está sendo validada.

## Conclusão

A metodologia atual busca equilibrar rigor experimental e simplicidade operacional:
- primeiro consolidamos baseline e FE estrutural;
- depois testamos geografia de forma controlada;
- só então avançamos para seleção de features, tunagem e avaliação econômica.

Essa ordem mantém a leitura dos resultados mais limpa e reduz o risco de concluir algo sobre `City` quando, na prática, o efeito veio de outra mudança no pipeline.

