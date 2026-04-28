# Estrutura Proposta da Experimentação

## Objetivo do documento

Consolidar a organização da etapa de experimentação do projeto, deixando explícito:

- qual é o papel de cada modelo;
- quais versões de dataset serão comparadas;
- em que momento cada técnica entra no fluxo;
- por que cada decisão metodológica foi tomada.

Este documento complementa o material já descrito em [experimentacao.md](C:/Users/jooar/ds/portfolio/mle-f1-tech-challenge-fiap/docs/experimentacao.md), mas organiza a execução com foco no objetivo central do projeto: evoluir a `MLP` até uma versão final tecnicamente sólida e economicamente justificável, usando a `Regressão Logística` como baseline obrigatório e o melhor modelo de árvore como benchmark.

## Premissas do projeto

- A `MLP` é o modelo principal do projeto e a solução que deve ser aprofundada.
- A `Regressão Logística` é o baseline obrigatório.
- Um único modelo de árvore será mantido como benchmark técnico.
- O melhor modelo de árvore será escolhido entre `DecisionTree`, `RandomForest` e `XGBoost`.
- A métrica principal de seleção técnica será `PR-AUC`.
- `CLTV` não será usado como feature preditiva principal do modelo neste fluxo-base.
- `CLTV` será tratado como metadata de negócio para avaliação econômica e, em etapas posteriores, poderá ser testado como `sample_weight`.

## Papel de cada modelo

### Regressão Logística

É o baseline obrigatório do projeto. Seu papel é oferecer uma referência simples, estável e interpretável para medir o quanto a `MLP` realmente agrega.

### MLP

É o modelo-alvo do projeto. Toda a estrutura experimental foi desenhada para permitir sua comparação justa com o baseline e com um benchmark forte, mas mantendo a `MLP` como foco principal de evolução.

### Melhor modelo de árvore

Serve como benchmark técnico. Não é necessariamente o modelo que será levado à produção, mas funciona como régua externa para responder se a `MLP` está competitiva frente a uma família de modelos tabulares tradicionalmente forte.

## Estratégia geral da experimentação

A experimentação será organizada em blocos sequenciais, de forma a isolar os efeitos de cada decisão:

1. medir sinal preditivo inicial no dataset original;
2. escolher o benchmark de árvore;
3. comparar os três modelos principais sob as mesmas condições;
4. introduzir feature engineering de forma controlada;
5. testar separadamente o efeito do `Churn Score`;
6. aplicar seleção de features com `SelectKBest`;
7. realizar tunagem com foco principal na `MLP`;
8. conduzir análise econômica com `CLTV`.

Essa ordem foi escolhida para evitar misturar, na mesma rodada, ganhos vindos de arquitetura, novas features, score legado, seleção de variáveis e critério econômico.

## Por que não seguir com todos os modelos até o final

Não faz sentido aprofundar todos os modelos igualmente porque esse não é o objetivo do projeto. O projeto não busca descobrir qualquer modelo vencedor para deployment. O objetivo é defender tecnicamente a melhor `MLP` possível.

Por isso:

- a `Regressão Logística` permanece por ser baseline obrigatório;
- a `MLP` permanece por ser o foco do projeto;
- apenas o melhor modelo de árvore permanece como benchmark.

Essa decisão reduz custo experimental sem enfraquecer a comparação.

## Protocolo experimental comum

As seguintes regras devem ser mantidas em todas as etapas:

- mesmo split holdout final;
- mesma estratégia de validação cruzada estratificada;
- mesma métrica principal (`PR-AUC`);
- mesma lógica de comparação entre folds;
- mesmas premissas de negócio ao longo da análise.

Isso é importante para que as diferenças observadas sejam atribuídas ao experimento em si, e não à mudança de protocolo.

## Estrutura em fases

### Fase 1 - Baseline inicial no dataset original

#### Objetivo

Verificar se há sinal preditivo útil no dataset original e estabelecer as primeiras referências comparativas.

#### Modelos avaliados

- `DummyClassifier`
- `LogisticRegression`
- `DecisionTree`
- `RandomForest`
- `XGBoost`
- `MLP`

#### O que essa fase responde

- existe sinal preditivo no dataset original;
- a `MLP` já nasce competitiva ou não;
- qual árvore deve seguir como benchmark;
- o problema parece mais linear ou não linear.

#### Saída esperada

- tabela comparativa inicial;
- escolha do melhor modelo de árvore;
- manutenção de três trilhas principais:
  - `LogisticRegression`
  - `MLP`
  - melhor modelo de árvore

#### Por que essa decisão foi tomada

Essa fase é necessária para não iniciar a evolução da `MLP` sem contexto. Antes de adicionar complexidade, é importante saber onde a `MLP` começa em relação ao baseline e a um benchmark forte.

### Fase 2 - Dataset com feature engineering estrutural

#### Objetivo

Adicionar features derivadas da EDA e medir ganho técnico sem misturar esse efeito com score legado de outro modelo.

#### Inclui

- features derivadas estruturais;
- transformações justificadas pela EDA;
- exclusão de `Churn Score` nesta fase;
- exclusão do tratamento especial de `City` nesta fase principal.

#### Não inclui

- `Churn Score`;
- target encoding de `City`.

#### O que essa fase responde

- as hipóteses da EDA realmente melhoram o poder preditivo;
- o ganho de feature engineering aparece de forma consistente nos três modelos principais;
- a `MLP` aproveita melhor as novas features do que o baseline.

#### Por que essa decisão foi tomada

O `Churn Score` representa empilhamento de um modelo anterior, e por isso distorce a leitura do impacto do feature engineering puro. Já `City` com target encoding exige controle adicional de leakage e, por isso, deve ser tratado em etapa separada ou posterior.

### Fase 3 - Dataset com feature engineering + Churn Score

#### Objetivo

Medir o efeito de adicionar `Churn Score` como uma fonte extra de informação ao conjunto de features.

#### Modelos avaliados

- `LogisticRegression`
- `MLP`
- melhor árvore

#### O que essa fase responde

- quanto o `Churn Score` eleva a performance em relação à fase anterior;
- se a `MLP` continua competitiva quando o problema passa a incorporar stacking;
- se o ganho técnico justifica a complexidade adicional.

#### Por que essa decisão foi tomada

`Churn Score` não deve ser tratado como feature comum. Ele altera a natureza do experimento, porque introduz a saída de outro modelo no processo. Por isso, seu efeito precisa ser medido separadamente.

### Fase 4 - Seleção de features com SelectKBest

#### Objetivo

Encontrar subconjuntos de features mais informativos maximizando `PR-AUC`.

#### Estratégia

Aplicar `GridSearchCV` com `SelectKBest` usando:

- `f_classif`
- `mutual_info_classif`

com busca orientada por `PR-AUC`.

#### Modelos avaliados

- `LogisticRegression`
- `MLP`
- melhor árvore

#### Observação importante

Essa etapa deve ocorrer dentro de `Pipeline`, para garantir que a seleção de features aconteça dentro de cada fold e evitar leakage.

#### O que essa fase responde

- qual valor de `k` funciona melhor em cada trilha;
- qual critério de seleção de features é mais útil;
- se reduzir dimensionalidade ajuda a `MLP` técnica e computacionalmente.

#### Por que essa decisão foi tomada

Fazer `SelectKBest` fora do pipeline aumentaria muito a complexidade da implementação da `MLP` e abriria espaço para erros metodológicos. Dentro do `Pipeline`, a etapa fica automatizada e corretamente encapsulada na validação cruzada.

### Fase 5 - Tratamento especial de City

#### Objetivo

Avaliar a variável `City` separadamente por meio de target encoding, caso o time decida seguir com essa hipótese.

#### Observação importante

Essa fase é separada porque `City` com target encoding tem alto risco de leakage se for implementada sem cuidado.

#### Requisito metodológico

O encoding precisa ser feito de forma segura dentro do protocolo de validação, idealmente com lógica `out-of-fold` na etapa de treino.

#### Por que essa decisão foi tomada

`City` é uma feature potencialmente útil, mas metodologicamente sensível. Separá-la evita contaminar a leitura das demais melhorias de feature engineering.

### Fase 6 - Tunagem

#### Objetivo

Encontrar a melhor configuração da `MLP` após definição da melhor versão técnica do dataset e da melhor estratégia de seleção de features.

#### Prioridade de esforço

- tunagem principal: `MLP`
- tunagem secundária e mais leve: `LogisticRegression` e benchmark de árvore

#### O que essa fase responde

- qual é a melhor `MLP` técnica do projeto;
- qual ganho adicional a tunagem entrega sobre a `MLP` base.

#### Por que essa decisão foi tomada

Como a `MLP` é o foco do projeto, ela deve concentrar o maior esforço de refinamento. Os demais modelos devem ser ajustados apenas o suficiente para garantir comparação honesta.

### Fase 7 - Análise econômica com CLTV

#### Objetivo

Avaliar o desempenho dos modelos sob a ótica de negócio usando `CLTV` como metadata econômica.

#### Comparações mínimas esperadas

- `LogisticRegression` finalista;
- `MLP` tunada;
- melhor modelo de árvore finalista.

#### Uso do CLTV nesta fase

- como metadata de avaliação econômica;
- para análise de impacto financeiro por priorização de clientes;
- opcionalmente, em experimento posterior, como `sample_weight`.

#### Por que essa decisão foi tomada

O `CLTV` carrega valor econômico e não deve ser misturado automaticamente com o conjunto de features do modelo. Primeiro ele deve ser usado para avaliar a qualidade econômica da priorização; depois, em uma rodada específica, pode ser testado como peso de treino.

## Por que o CLTV não entra diretamente como feature neste fluxo

Essa decisão foi tomada por três razões:

1. o objetivo principal é avaliar o modelo preditivo e a política econômica de forma separada;
2. usar `CLTV` como metadata facilita interpretar retorno financeiro sem contaminar a representação preditiva;
3. isso permite comparar, de forma mais limpa, o ganho vindo do modelo com o ganho vindo da estratégia de negócio.

Além disso, manter `CLTV` inicialmente fora do conjunto de features evita que o modelo aprenda atalhos que podem não refletir exatamente o objetivo de priorização.

## Por que usar Pipeline

Mesmo com a maior complexidade da `MLP`, o uso de `Pipeline` segue sendo desejável por três motivos:

1. garante que transformações como feature engineering, encoding e seleção de features sejam aplicadas dentro de cada fold;
2. reduz risco de leakage;
3. simplifica a automação da busca por `k` no `SelectKBest`.

No caso da `MLP`, isso aponta para a criação de um wrapper compatível com a interface do `scikit-learn`.

## Por que transformar feature_engineering.py em transformer

Essa decisão melhora a qualidade experimental porque:

- encapsula a lógica de criação de features;
- garante aplicação consistente dentro da validação cruzada;
- reduz código procedural no notebook;
- permite reuso em `Pipeline` com `GridSearchCV`.

Ou seja, além de organizar melhor o código, essa escolha fortalece a validade metodológica dos experimentos.

## Por que tratar City separadamente

`City` é um caso especial porque seu encoding supervisionado depende do alvo. Isso gera risco real de leakage se a média histórica da categoria for calculada de forma ingênua.

Portanto:

- `City` não deve entrar misturado com as demais features no primeiro bloco principal de FE;
- seu uso deve ser tratado como experimento controlado;
- a implementação deve ser cuidadosamente desenhada.

## Por que manter a análise econômica depois da definição técnica principal

A análise econômica não deve substituir o baseline técnico, mas também não deve ficar invisível até o último momento. Por isso, a organização proposta preserva a lógica:

- primeiro estabilizar a comparação técnica;
- depois medir valor econômico sobre os candidatos mais relevantes.

Essa abordagem evita gastar energia em análise financeira de modelos que já se mostraram tecnicamente pouco competitivos, sem deixar a dimensão de negócio apenas como pós-escrito.

## Matriz resumida dos experimentos

| Fase | Dataset / Estratégia | Modelos | Objetivo principal |
| --- | --- | --- | --- |
| 1 | Dataset original | Dummy, LogReg, DT, RF, XGB, MLP | Escolher baseline, medir sinal inicial e definir benchmark de árvore |
| 2 | Dataset original + FE estrutural | LogReg, MLP, melhor árvore | Medir ganho de FE puro |
| 3 | FE estrutural + Churn Score | LogReg, MLP, melhor árvore | Medir efeito de stacking |
| 4 | Fase 2 e/ou 3 + SelectKBest | LogReg, MLP, melhor árvore | Encontrar melhor subconjunto de features por PR-AUC |
| 5 | Experimento separado com City | LogReg, MLP, melhor árvore | Medir impacto de target encoding controlado |
| 6 | Melhor configuração técnica | MLP prioritariamente | Tunar a MLP |
| 7 | Avaliação com CLTV | LogReg, MLP, melhor árvore | Comparar desempenho econômico |

## Encaminhamento recomendado

Como próximo passo de implementação, a ordem sugerida é:

1. transformar o `feature_engineering.py` em transformer compatível com sklearn;
2. manter `City` fora da primeira versão do transformer;
3. criar o wrapper da `MLP` para uso em `Pipeline`;
4. montar pipeline com `FeatureEngineeringTransformer`, `preprocessor`, `SelectKBest` e estimador;
5. rodar a fase 1 para escolher o benchmark de árvore;
6. seguir com as fases restantes apenas com `LogReg`, `MLP` e benchmark escolhido.

## Conclusão

A estrutura proposta busca equilibrar quatro necessidades do projeto:

- rigor metodológico;
- controle de complexidade de implementação;
- alinhamento com o objetivo central de entregar a melhor `MLP`;
- capacidade de justificar decisões técnicas e econômicas de forma clara.

O desenho final evita leakage, reduz ambiguidade interpretativa e mantém foco no que de fato precisa ser defendido pelo time: se a `MLP`, quando bem estruturada, realmente supera o baseline e se mantém competitiva frente a um benchmark forte, além de gerar valor econômico mensurável.
