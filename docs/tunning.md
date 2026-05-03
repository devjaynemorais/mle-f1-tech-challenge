# Hyperparameter Tunning

## Objetivo

Esta etapa tem como objetivo otimizar os hiperparâmetros dos modelos finais após a definição do espaço de features.

No fluxo atual, a ordem metodológica adotada é:

1. Definir um baseline comparável
2. Adicionar feature engineering
3. Validar o ganho de `Churn Score`
4. Testar e descartar variáveis geográficas para o pipeline final
5. Aplicar feature selection
6. Realizar hyperparameter tuning sobre a configuração vencedora

Com isso, o tuning passa a atuar em um espaço de entrada já estabilizado:

- features de negócio derivadas pelo `FeatureEngineerTransformer`
- `Churn Score` mantido
- geografia removida com `GeoTransformer(strategy="drop")`
- seleção de features fixa em `SelectKBest(score_func=f_classif, k=18)` para o `XGBoost`
- seleção de features fixa em `SelectKBest(score_func=mutual_info_classif, k=47)` para a `MLP`

## Metodologia

Para `XGBoost` e `MLP`, a busca é feita por uma grade exaustiva com orçamento de tempo controlado.

Princípios adotados:

- métrica principal de seleção: `recall`
- desempate secundário: `pr_auc`
- validação: `cross_validate` com o mesmo protocolo de CV do notebook
- orçamento máximo: `30 minutos`
- parada por tempo: se o orçamento for atingido, a busca é encerrada e a melhor configuração encontrada até aquele ponto é mantida

Esse desenho foi escolhido porque:

- `recall` é a métrica prioritária do problema
- `XGBoost` possui espaço de busca amplo e custo computacional mais alto
- a `MLP` também exige custo relevante por conta do treino iterativo com early stopping
- um `GridSearchCV` puro não oferece controle natural de budget por tempo
- a busca manual permite interromper com segurança sem perder o melhor resultado já observado

## Estrutura do pipeline

O pipeline avaliado no tuning do `XGBoost` segue a ordem:

1. `FeatureEngineerTransformer(**round4_fe_params)`
2. `GeoTransformer(strategy="drop")`
3. `preprocessor`
4. `SelectKBest(score_func=f_classif, k=18)`
5. `XGBClassifier(...)`

Essa estrutura preserva consistência com a conclusão das rodadas anteriores:

- manter apenas as features aprovadas no fluxo experimental
- retirar geografia do pipeline final
- reduzir dimensionalidade antes do modelo

O pipeline avaliado no tuning da `MLP` segue a ordem:

1. `FeatureEngineerTransformer(**round4_fe_params)`
2. `GeoTransformer(strategy="drop")`
3. `preprocessor`
4. `SelectKBest(score_func=mutual_info_classif, k=47)`
5. `StandardScaler(with_mean=False)`
6. `MLPClassifierWrapper(...)`

Essa estrutura preserva compatibilidade com a rede já usada nas rodadas anteriores, mas fixa o melhor espaço de entrada encontrado para a `MLP`.

## Hiperparâmetros testados no XGBoost

### `n_estimators`

Controla o número de árvores construídas.

Impacto:

- valores maiores aumentam capacidade do modelo
- também elevam custo computacional
- podem causar overfitting se não compensados por learning rate e regularização

### `max_depth`

Controla a profundidade máxima de cada árvore.

Impacto:

- profundidades maiores capturam interações mais complexas
- aumentam risco de overfitting
- deixam o modelo mais sensível a ruído

### `learning_rate`

Controla o tamanho do passo em boosting.

Impacto:

- valores menores tornam o aprendizado mais estável
- normalmente exigem mais árvores
- valores maiores aceleram treinamento, mas podem piorar generalização

### `min_child_weight`

Controla o volume mínimo de informação exigido para criar novos splits.

Impacto:

- valores maiores deixam a árvore mais conservadora
- ajudam a reduzir splits frágeis
- são úteis para controlar overfitting

### `subsample`

Define a fração de observações usada por árvore.

Impacto:

- valores menores introduzem aleatoriedade
- podem melhorar generalização
- valores muito baixos podem reduzir estabilidade

### `colsample_bytree`

Define a fração de colunas usada por árvore.

Impacto:

- reduz correlação entre árvores
- pode melhorar robustez
- também controla a complexidade do ensemble

### `gamma`

Exige ganho mínimo para que um split seja aceito.

Impacto:

- valores maiores tornam o modelo mais conservador
- reduzem splits pouco úteis
- ajudam a podar interações fracas

### `reg_alpha`

Regularização L1 dos pesos das folhas.

Impacto:

- força soluções mais esparsas
- pode reduzir complexidade
- é útil quando há risco de excesso de sensibilidade a ruído

### `reg_lambda`

Regularização L2 dos pesos das folhas.

Impacto:

- estabiliza os pesos
- reduz variância
- costuma ajudar a controlar overfitting de forma suave

## Hiperparâmetros testados na MLP

Nesta rodada, a `MLP` foi aberta de forma conservadora. A arquitetura continua com:

- uma única camada oculta
- saída binária com um logit
- mesmos defaults já usados anteriormente no projeto

Defaults preservados:

- `activation="relu"`
- `hidden_dim=64`
- `batch_size=64`
- `lr=1e-3`
- `weight_decay=1e-5`
- `max_epochs=80`
- `patience=8`
- `threshold=0.5`

Importante:

- `CityEmbeddingMLP` não entra neste tuning
- `dropout` não entra nesta rodada
- a mudança de arquitetura foi feita com foco em compatibilidade e não em redes mais profundas

### `activation`

Define a função de ativação da camada oculta.

Opções suportadas:

- `relu`
- `leaky_relu`
- `elu`
- `gelu`
- `tanh`

Impacto:

- controla a forma da não linearidade
- influencia fluxo de gradiente e estabilidade do treino
- pode alterar sensibilidade da rede a escalas e regiões saturadas

### `hidden_dim`

Controla o número de neurônios da camada oculta.

Impacto:

- valores maiores aumentam a capacidade da rede
- podem capturar relações mais complexas
- também aumentam risco de overfitting e custo computacional

### `lr`

Taxa de aprendizado do otimizador.

Impacto:

- valores altos aceleram o treino, mas podem causar instabilidade
- valores baixos tendem a ser mais estáveis, porém mais lentos

### `batch_size`

Tamanho do mini-batch no treino.

Impacto:

- batches menores introduzem mais ruído no gradiente
- batches maiores podem estabilizar a atualização e melhorar throughput
- também alteram o tempo por época

### `weight_decay`

Regularização L2 aplicada no otimizador.

Impacto:

- reduz complexidade efetiva dos pesos
- ajuda a conter overfitting
- normalmente funciona como regularização principal em redes pequenas

### `patience`

Controla o early stopping.

Impacto:

- valores menores encerram o treino mais cedo
- valores maiores permitem explorar mais épocas antes de parar
- afeta diretamente o equilíbrio entre custo computacional e convergência

### `threshold`

Controla o limiar de decisão usado para transformar probabilidade em classe.

Impacto:

- não altera o treino da rede
- altera o trade-off entre `recall` e `precision`
- é especialmente importante quando o objetivo principal é maximizar `recall`

## Sobre o dropout

O `dropout` não foi incluído nesta rodada.

Motivo:

- a rede atual é pequena e rasa
- ainda não há evidência forte de overfitting que justifique esse aumento de complexidade
- nesta configuração, `weight_decay` já cobre a necessidade mais imediata de regularização

Em termos práticos, o `dropout` atua desligando aleatoriamente parte dos neurônios durante o treino. Isso reduz coadaptação entre unidades e pode melhorar generalização, mas também pode:

- deixar o treino mais lento
- dificultar convergência
- atrapalhar modelos pequenos quando o overfitting não é um problema dominante

## Critério de seleção final

A melhor configuração é definida por:

1. maior `recall_mean`
2. em caso de empate, maior `pr_auc_mean`

Esse critério foi escolhido para alinhar a busca com o objetivo do problema, priorizando a recuperação de casos positivos sem perder completamente a noção de qualidade probabilística.

## Saída esperada

Ao final da busca, a rotina deve registrar:

- hiperparâmetros testados
- métricas médias de CV
- tempo médio de treino e score
- melhor configuração encontrada até o momento
- melhor estimador ajustado no conjunto completo

Se o limite de `30 minutos` for atingido antes do fim da grade, o experimento ainda produz um resultado válido, retornando o melhor modelo encontrado dentro do budget.
