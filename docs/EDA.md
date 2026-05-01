# EDA - Storytelling Analítico do Churn

## Objetivo

Este documento consolida a análise exploratória realizada em [`notebooks/01_exploratory_data_analysis.ipynb`](/C:/Users/jooar/ds/portfolio/mle-f1-tech-challenge-fiap/notebooks/01_exploratory_data_analysis.ipynb) e transforma os outputs do notebook em uma leitura de negócio orientada a decisão.

O foco da EDA foi responder três perguntas:

1. Onde o churn está mais concentrado.
2. Quais variáveis carregam sinal preditivo real.
3. Quais hipóteses valem virar feature engineering na próxima etapa.

## Resumo Executivo

A história principal do dataset é bastante clara: o churn não acontece de forma aleatória. Ele se concentra em clientes novos, com contratos flexíveis, internet fibra, menor proteção de suporte e maior fricção de pagamento.

O desenho do problema sugere um funil de risco:

1. O cliente entra com pouco tempo de casa.
2. Permanece em contrato `Month-to-month`.
3. Usa `Fiber optic`, que carrega churn muito acima da média.
4. Não contrata camadas de proteção, como `Online Security` e `Tech Support`.
5. Paga via `Electronic check`, sem automatização.

Quando esse conjunto aparece junto, o risco de churn cresce de forma relevante. No lado oposto, clientes antigos, com contrato longo, pagamento automático e serviços de suporte ativados formam o grupo mais estável.

## Base e Qualidade dos Dados

### Visão geral

- Base com `7.043` linhas e `33` colunas.
- Apenas `Churn Reason` apresenta missing estrutural relevante: `73,46%`.
- Após o tratamento, `Total Charges` passou a `float` com `7.032` não nulos, indicando `11` registros faltantes nessa variável.
- Não há registros duplicados.

### Leitura de qualidade

- `Country` e `State` são constantes em toda a base (`United States` e `California`), então não agregam discriminação para o modelo.
- `City` possui `1.129` categorias, o que confirma alta cardinalidade.
- `Zip Code`, `Latitude`, `Longitude` e `Lat Long` também têm alta cardinalidade e funcionam mais como identificadores geográficos do que como boas features brutas.
- `Churn Reason` é pós-evento e não pode entrar no modelo preditivo.
- `Churn Score` e `CLTV` precisam de cautela, pois podem carregar informação derivada e gerar leakage.

## Divisão das Variáveis no Notebook

O notebook organizou as variáveis nos seguintes grupos.

### Variáveis geográficas

- `Country`
- `State`
- `City`
- `Zip Code`
- `Latitude`
- `Longitude`
- `Lat Long`

### Variáveis demográficas

- `Gender`
- `Senior Citizen`
- `Partner`
- `Dependents`

### Variáveis de relacionamento com o cliente

- `Tenure Months`
- `Contract`
- `Paperless Billing`
- `Payment Method`

### Variáveis de serviços contratados

- `Phone Service`
- `Multiple Lines`
- `Internet Service`
- `Online Security`
- `Online Backup`
- `Device Protection`
- `Tech Support`
- `Streaming TV`
- `Streaming Movies`

### Variáveis financeiras

- `Monthly Charges`
- `Total Charges`

### Variáveis derivadas / score

- `Churn Score`
- `CLTV`

### Variáveis relacionadas ao churn e pós-evento

- `Churn Label`
- `Churn Value`
- `Churn Reason`

### Variáveis técnicas / identificação

- `CustomerID`
- `Count`

### Recorte analítico usado nas análises bivariadas

Depois da limpeza inicial, o notebook trabalhou principalmente com:

- Numéricas: `Tenure Months`, `Monthly Charges`, `Total Charges`, `Churn Value`, `Churn Score`, `CLTV`
- Categóricas: `City`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`

## Storytelling das Análises

### 1. Variáveis geográficas

O nível país/estado não explica o churn, porque toda a base está concentrada no mesmo recorte: `United States` e `California`.

O sinal geográfico aparece em `City`, mas de forma bruta ela é inviável para modelagem direta:

- `City` possui `1.129` categorias.
- O agrupamento regional feito no notebook mostrou taxas de churn diferentes:
  - `socal`: `29,89%`
  - `central`: `27,78%`
  - `norcal`: `26,99%`
  - `other`: `26,18%`

Além disso, entre cidades com maior volume, apareceram taxas elevadas em `San Diego` (`33,3%`), `Glendale` (`32,5%`), `Pasadena` (`30,0%`), `San Francisco` (`29,8%`) e `Los Angeles` (`29,5%`).

**Conclusão:** existe sinal geográfico, mas ele precisa ser regularizado via clusterização, frequência ou target encoding. `City` não deve entrar crua.

### 2. Variáveis demográficas e familiares

As análises bivariadas mostraram pouca diferença entre homens e mulheres:

- `Female`: `26,92%`
- `Male`: `26,16%`

Já as variáveis ligadas à estrutura familiar apresentam sinal forte:

- `Senior Citizen = Yes`: `41,68%`
- `Senior Citizen = No`: `23,61%`
- `Partner = No`: `32,96%`
- `Partner = Yes`: `19,66%`
- `Dependents = No`: `32,55%`
- `Dependents = Yes`: `6,52%`

**Conclusão:** `Gender` tende a ter baixo valor preditivo. Em contrapartida, `Dependents`, `Partner` e, em menor grau, `Senior Citizen`, ajudam a capturar estabilidade do vínculo com a operadora.

### 3. Tipos de serviços

Aqui aparece um dos sinais mais fortes do dataset.

#### Internet e serviços de proteção

- `Internet Service = Fiber optic`: `41,89%`
- `Internet Service = DSL`: `18,96%`
- `Internet Service = No`: `7,40%`

- `Online Security = No`: `41,77%`
- `Online Security = Yes`: `14,61%`

- `Tech Support = No`: `41,64%`
- `Tech Support = Yes`: `15,17%`

- `Online Backup = No`: `39,93%`
- `Online Backup = Yes`: `21,53%`

- `Device Protection = No`: `39,13%`
- `Device Protection = Yes`: `22,50%`

#### Serviços de consumo e telefonia

- `Phone Service` praticamente não muda o churn: `26,71%` vs `24,93%`
- `Multiple Lines` muda pouco: `28,61%` no `Yes` contra `25,04%` no `No`
- `Streaming TV` e `Streaming Movies` mostram diferença, mas bem menor do que as variáveis de suporte

**Conclusão:** o churn está muito mais associado à ausência de camadas de proteção e suporte do que a serviços de entretenimento ou telefonia em si. Isso reforça a hipótese de que o problema mistura sensibilidade a preço com experiência de serviço.

### 4. Variáveis financeiras

#### Contrato e faturamento

O tipo de contrato é um dos principais separadores do churn:

- `Month-to-month`: `42,71%`
- `One year`: `11,27%`
- `Two year`: `2,83%`

Também existe diferença relevante por meio de pagamento:

- `Electronic check`: `45,29%`
- `Mailed check`: `19,11%`
- `Bank transfer (automatic)`: `16,71%`
- `Credit card (automatic)`: `15,24%`

E a cobrança digital também sinaliza risco:

- `Paperless Billing = Yes`: `33,57%`
- `Paperless Billing = No`: `16,33%`

**Conclusão:** churn é muito mais alto quando o cliente tem saída fácil, baixa automação de pagamento e menor custo de troca. `Contract` e `Payment Method` são candidatos naturais a features fortes.

#### Receita e monetização

Os gráficos e boxplots do notebook indicam duas leituras principais:

- `Monthly Charges` se relaciona positivamente com churn, mas de forma não linear.
- `Tenure Months` funciona como variável de proteção: conforme o cliente envelhece na base, o risco tende a cair.

O output descritivo reforça isso:

- `Tenure Months`: mediana `29`, máximo `72`
- `Monthly Charges`: mediana `70,35`
- `Total Charges`: mediana `1.397,48`

**Conclusão:** a relação entre preço e churn depende do estágio do cliente. Preço alto pesa mais em clientes recentes do que em clientes maduros.

### 5. CLTV, Churn Score e churn reasons

#### CLTV

`CLTV` tem alta cardinalidade (`3.438` valores únicos) e forte multicolinearidade com outras numéricas. Como a estratégia do projeto não usa `CLTV` como feature do modelo, ele deve permanecer fora do pipeline preditivo principal e ser tratado como metadata de negócio.

#### Churn Score

`Churn Score` também mostrou alta cardinalidade (`85` valores únicos) e entrou entre as variáveis numéricas com VIF elevado. Ele pode carregar sinal útil, mas deve ser tratado como experimento controlado, nunca como feature "automática" de baseline.

#### Churn reasons

Os motivos mais frequentes de churn reforçam a narrativa de negócio:

- Concorrência respondeu por aproximadamente `33,2%` dos casos, somando:
  - `Competitor offered higher download speeds`
  - `Competitor offered more data`
  - `Competitor made better offer`
  - `Competitor had better devices`
- Problemas ligados a suporte e atendimento responderam por cerca de `19,6%`, somando:
  - `Attitude of support person`
  - `Attitude of service provider`
  - `Poor expertise of phone support`
  - `Poor expertise of online support`

**Conclusão:** o churn parece nascer da combinação entre pressão competitiva, percepção de valor e experiência de suporte.

## Leitura dos Testes do Notebook

### Qui-quadrado + V de Cramer

O teste de associação para variáveis categóricas foi um dos outputs mais consistentes do notebook.

As variáveis com associação mais forte com churn foram:

- `City`: `Cramer_V = 0,4185`
- `Contract`: `0,4101`
- `Online Security`: `0,3474`
- `Tech Support`: `0,3429`
- `Internet Service`: `0,3225`
- `Payment Method`: `0,3034`

No outro extremo:

- `Phone Service`: não significativo (`p = 0,3388`)
- `Gender`: não significativo (`p = 0,4866`)
- `Multiple Lines`: significativo, mas muito fraco (`Cramer_V = 0,0401`)

**Conclusão:** o núcleo do churn está em contrato, internet, suporte, pagamento e geografia. `Gender` e `Phone Service` podem ser tratados como baixa prioridade.

### Pairplot e matriz de correlação

Visualmente, os gráficos reforçaram três padrões:

- `Tenure Months` tem relação inversa com churn.
- `Monthly Charges` separa churners em patamar mais alto de cobrança.
- `Total Charges` mistura efeito de permanência e gasto acumulado, então não deve ser lido isoladamente.

**Conclusão:** o problema tem sinal, mas não parece puramente linear. Interações e transformações monotônicas devem ajudar.

### VIF - multicolinearidade

Os maiores VIFs foram:

- `Total Charges`: `14,74`
- `Tenure Months`: `13,75`
- `Monthly Charges`: `13,68`
- `CLTV`: `11,51`
- `Churn Score`: `11,32`

**Conclusão:** as variáveis numéricas carregam redundância forte. Isso favorece regularização, seleção de features e versões transformadas em vez de simplesmente empilhar tudo no modelo.

### Information Value (IV)

Após a correção da fórmula do `IV` na classe `AnaliseIV`, os resultados passaram a ficar coerentes com os demais testes do notebook. O problema anterior estava no sinal do `IV` para a convenção `good_over_bad`, o que invertia os valores agregados e fazia praticamente tudo parecer `Irrelevante`.

Com o cálculo corrigido, o `IV` confirma o mesmo núcleo de variáveis já apontado por taxa de churn, qui-quadrado e árvore:

- `City`: `2,253`
- `Contract`: `1,239`
- `Online Security`: `0,718`
- `Tech Support`: `0,700`
- `Internet Service`: `0,618`
- `Online Backup`: `0,529`
- `Device Protection`: `0,500`
- `Dependents`: `0,459`
- `Payment Method`: `0,457`

Na base da fila, os valores continuam compatíveis com pouca utilidade preditiva:

- `Multiple Lines`: `0,008`
- `Phone Service`: `0,001`
- `Gender`: aproximadamente `0,000`

**Conclusão:** o `IV` voltou a ser útil como evidência complementar de priorização. Ainda assim, ele não deve ser lido sozinho. Em especial, `City` merece cautela: o valor muito alto é consistente com a presença de sinal, mas também pode ser inflado pela cardinalidade extrema e por categorias raras. A leitura correta é que geografia importa, mas precisa entrar no modelo de forma regularizada.

### Árvore de decisão para importância exploratória

As importâncias mais altas foram:

1. `Tenure Months` - `0,3719`
2. `Internet Service_Fiber optic` - `0,2903`
3. `Dependents_Yes` - `0,0854`
4. `Monthly Charges` - `0,0409`
5. `Total Charges` - `0,0408`
6. `Contract_Two year` - `0,0224`
7. `Payment Method_Electronic check` - `0,0222`

O próprio notebook resumiu os drivers desta forma:

- Tier 1: `Tenure Months`, `Contract`, `Internet Service`, `Online Security`, `Tech Support`
- Tier 2: `Dependents`, `Payment Method`, `Online Backup`, `Device Protection`
- Tier 3: `Monthly Charges`, `Paperless Billing`, `Partner`, `Senior Citizen`
- Tier 4: `Gender`, `Phone Service`, `Multiple Lines`

**Conclusão:** a árvore confirma o storytelling da EDA e ajuda a ordenar o backlog de features.

## Principais Conclusões

- A taxa atual de churn da base é `26,54%`.
- O risco de churn é decrescente no tempo; clientes novos concentram a maior fragilidade.
- `Contract` é uma das variáveis mais fortes do problema: contratos longos reduzem drasticamente o churn.
- `Fiber optic` aparece fortemente associada ao churn, especialmente quando o cliente não possui serviços de proteção e suporte.
- `Payment Method` diferencia grupos de risco com clareza, principalmente `Electronic check`.
- `Dependents` e `Partner` funcionam como proxies de estabilidade de vínculo.
- `Gender` e `Phone Service` têm baixo valor explicativo e podem ficar em baixa prioridade.
- Com o `IV` corrigido, `Contract`, `Online Security`, `Tech Support`, `Internet Service`, `Payment Method` e `Dependents` ganham reforço estatístico adicional.
- `City` carrega sinal, mas exige tratamento por alta cardinalidade.
- `CLTV` não deve entrar como feature do modelo neste fluxo; seu papel é econômico e não preditivo.
- `Churn Score` pode ser testado em ablação controlada, por risco de leakage.
- Há multicolinearidade importante entre as variáveis numéricas, então seleção e transformação devem fazer parte da fase seguinte.

## Hipóteses para Testar em Feature Engineering

1. O efeito de `Tenure Months` é não linear, então discretizações e transformações logarítmicas devem performar melhor do que a variável crua isolada.
2. O risco de `Fiber optic` aumenta quando faltam camadas de proteção como `Online Security`, `Tech Support`, `Online Backup` e `Device Protection`.
3. A combinação `Month-to-month` + `Electronic check` representa um perfil de churn muito mais forte do que cada variável sozinha.
4. Sinais familiares, como `Partner` e `Dependents`, podem ser melhor capturados por uma feature sintética de estabilidade do domicílio.
5. O cliente não cancela apenas por preço alto; ele cancela quando preço alto aparece cedo na jornada, então interações entre `Monthly Charges` e `Tenure Months` tendem a ajudar.
6. O sinal geográfico existe, mas precisa de regularização; encoding de cidade deve funcionar melhor do que one-hot bruto.
7. Serviços de suporte devem gerar ganho maior do que serviços de entretenimento, então scores focados em proteção tendem a ser mais úteis do que scores gerais de consumo.
8. Combinações de fricção contratual e operacional, como `Month-to-month` + `Electronic check` ou `Paperless Billing` + `Electronic check`, tendem a capturar um risco mais forte do que cada coluna isolada.
9. O `IV` muito alto de `City` sugere testar não apenas encoding contínuo, mas também bandas de risco geográfico para reduzir sensibilidade a categorias raras.
10. `Churn Score` pode elevar performance aparente, mas talvez de forma artificial; precisa ser testado em rodada separada do baseline.

## Features Candidatas para a Próxima Fase

Lista consolidada com aproximadamente 15 features sugeridas para experimentação. Com o `IV` corrigido, a priorização abaixo ficou mais orientada pelas variáveis que agora aparecem fortes de maneira consistente em múltiplos testes. A feature `high_risk_high_value` ficou de fora porque depende de `CLTV`, e `CLTV` não faz parte do conjunto preditivo deste fluxo.

| Feature | Status | Racional da EDA | Observação |
| --- | --- | --- | --- |
| `contract_ordinal` | Já prevista | `Contract` é fortíssimo em taxa de churn, árvore, qui-quadrado e IV | Mapping atual: `Month-to-month=2`, `One year=1`, `Two year=0` |
| `tenure_group_ordinal` | Já prevista | Risco cai com o tempo e a relação é monotônica | Mapping atual: `new=2`, `mid=1`, `loyal=0` |
| `tenure_log` | Já prevista | Captura a não linearidade do tempo de casa | Boa candidata para modelos lineares |
| `city_target_rate` | Já prevista | `City` voltou a aparecer muito forte no IV, mas exige regularização | Fazer dentro do pipeline para evitar leakage |
| `city_region_cluster` | Nova | O agrupamento `socal/norcal/central/other` mostrou diferenças de churn | Alternativa mais interpretável ao target encoding |
| `city_risk_band` | Nova | O IV alto de `City` sugere testar faixas geográficas de risco mais estáveis | Ex.: top/mid/low risk cities por treino |
| `digital_protection_score` | Já prevista | `Online Security`, `Tech Support`, `Online Backup` e `Device Protection` aparecem fortes no IV | Preferir score focado em proteção, não em entretenimento |
| `support_gap_count` | Nova | Quantidade de proteções ausentes pode capturar risco acumulado | Contar quantos entre `Security`, `Support`, `Backup`, `Protection` estão em `No` |
| `fiber_no_support_flag` | Já prevista | Combinação de fibra com falta de suporte é crítica | Forte candidata para interação |
| `family_stability_flag` | Já prevista | `Partner` e `Dependents` indicam maior estabilidade | Feature simples e interpretável |
| `payment_automatic_flag` | Nova | Pagamentos automáticos concentram churn menor | `Bank transfer` ou `Credit card` |
| `electronic_check_flag` | Nova | `Electronic check` é o meio com maior taxa de churn e IV relevante em `Payment Method` | Pode ser melhor que manter só o one-hot bruto |
| `paperless_echeck_flag` | Nova | Une cobrança digital com pagamento manual, perfil mais volátil | Interação com cara de fricção operacional |
| `price_pressure_ratio` | Nova | Preço pesa mais no início da jornada | Exemplo: `Monthly Charges / (Tenure Months + 1)` |


## Direcionamento Final para a Fase de FE

Se precisarmos priorizar o backlog, a ordem sugerida pela EDA é:

1. Features de tempo e contrato.
2. Features de proteção e suporte.
3. Features de pagamento.
4. Features de estabilidade familiar.
5. Features geográficas regularizadas.
6. Features experimentais com risco de leakage, como `Churn Score`, apenas em ablação controlada.

Em termos práticos, o melhor próximo passo não é adicionar dezenas de variáveis novas de uma vez, mas montar blocos temáticos de teste e medir ganho incremental por família de feature.
