# Arquitetura de Deploy em Batch - Predição de Churn

## Visão Geral

Este documento descreve a arquitetura de deploy em batch adotada para o modelo de predição de churn de clientes de telecomunicações. A solução foi projetada para realizar scoring periódico (diário ou semanal) de toda a base de clientes, permitindo a identificação proativa de clientes com risco de cancelamento e a elaboração de campanhas de retenção direcionadas.

## Justificativa da Arquitetura Batch

### Natureza do Problema

A decisão de adotar inferência em batch ao invés de streaming em tempo real foi baseada nas seguintes características do problema de negócio:

1. **Processo de Churn é Lento**: O churn de clientes de telecomunicações não é um evento instantâneo. É um processo gradual que envolve insatisfação acumulada, comparação com concorrentes e decisão ponderada ao longo de semanas ou meses.

2. **Dependência de Histórico**: As features mais preditivas do modelo dependem de dados históricos consolidados:
   - **Tenure Months**: Tempo total de contrato (acumulado mensalmente)
   - **Total Charges**: Valor total gasto pelo cliente (acumulado histórico)
   - **CLTV (Customer Lifetime Value)**: Métrica calculada baseada no histórico de consumo
   - **Padrões de Consumo**: Serviços contratados ao longo do tempo

3. **Independência de Eventos em Tempo Real**: A predição de churn não requer reação imediata a eventos. Mesmo que um cliente tenha uma experiência negativa hoje, isso só se refletirá no risco de churn após um período de reflexão.

### Limitações Técnicas do Modelo MLP

O modelo Multilayer Perceptron (MLP) implementado possui características que o tornam mais adequado para batch:

1. **Dependência de Features Agregadas**: O MLP foi treinado com features que requerem agregação temporal e cálculos complexos (como CLTV e engajamento de serviços), que não estão disponíveis em tempo real.

2. **Pré-processamento Complexo**: O pipeline de features inclui:
   - Encoding ordinal de variáveis categóricas
   - Normalização/standardization de variáveis numéricas
   - Criação de features derivadas (ex: `Valuable_HighRisk`, `Engagement_Score`)
   - Mapeamento geográfico de cidades para regiões

3. **Batch Efficiency**: Redes neurais são mais eficientes quando processam lotes de dados, aproveitando operações vetoriais em GPU/CPU.

### Vantagens da Abordagem Batch

| Vantagem | Descrição |
|----------|-----------|
| **Simplicidade** | Arquitetura mais simples de implementar, testar e manter |
| **Custo Reduzido** | Não requer infraestrutura de streaming (Kafka, Flink, etc.) |
| **Processamento Otimizado** | Pode ser executado em horários de baixo custo computacional |
| **Consistência** | Garante que todas as predições usem a mesma versão do modelo |
| **Auditoria** | Facilita o rastreamento e reprodução de resultados |
| **Integração com BI** | Resultados podem ser facilmente exportados para dashboards e sistemas de CRM |

### Frequência de Execução

- **Frequência Ideal**: Diária (recomendada para detecção oportuna)
- **Frequência Aceitável**: Semanal (reduz custo computacional em 7x)

A frequência diária permite:
- Detecção mais rápida de mudanças no perfil de risco
- Atualização diária de dashboards de monitoramento
- Disparo de ações de retenção no mesmo dia da identificação

## Arquitetura da Solução

### Diagrama de Componentes

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Base de Dados │───▶│  Pipeline de     │───▶│   Modelo MLP    │
│   de Clientes   │    │  Feature Eng.    │    │   (PyTorch)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Sistema de CRM │◀───│  Resultados      │◀───│  Post-          │
│  / Campanhas    │    │  (Batch Output)  │    │  Processamento  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Componentes Principais

#### 1. Pipeline de Dados (`src/data/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `load_raw_data.py` | Carregamento dos dados brutos da fonte |
| `clean_data.py` | Limpeza e tratamento de valores missing |
| `make_dataset.py` | Transformações básicas e padronização |
| `split_data.py` | Divisão train/test para validação |

#### 2. Engenharia de Features (`src/features/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `feature_engineering.py` | Criação de features derivadas: |
| | - `Valuable_HighRisk`: Clientes valiosos com alto risco |
| | - `Engagement_Score`: Score baseado em serviços ativos |
| | - `Tenure_Group`: Categorização de tempo de contrato |
| | - `Contract_Ordinal`: Encoding ordinal por risco |
| | - `Family_Stability`: Flag de estabilidade familiar |
| | - `Fiber_No_Support`: Flag de fibra sem suporte técnico |
| `encoders.py` | Encoding de variáveis categóricas |
| `apply_feature_engineering.py` | Aplicação do pipeline de features |

#### 3. Modelo (`src/models/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `mlp.py` | Arquitetura MLP com PyTorch: |
| | - Camada de entrada (input_dim features) |
| | - Camada oculta (64 neurônios, ReLU) |
| | - Camada de saída (1 neurônio, sigmoid) |
| | - Treino com early stopping |
| `run_train.py` | Pipeline de treinamento com validação |
| `predict_model.py` | Script de inferência em batch |

**Arquitetura do MLP:**
```python
MLP(
  input_dim: número de features após engenharia
  hidden_dim: 64 neurônios
  output_dim: 1 (classificação binária)
)
```

#### 4. API de Inferência Batch (`src/api/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `main.py` | FastAPI com endpoints batch |
| `predictor.py` | Carregamento do modelo e inferência |
| `schemas.py` | Validação de entrada/saída com Pydantic |

**Endpoints:**
- `GET /health`: Health check do modelo
- `POST /predict`: Inferência batch (recebe lista de clientes, retorna predições)

#### 5. Configuração (`config/`)

| Arquivo | Responsabilidade |
|---------|------------------|
| `config.yaml` | Configuração principal do pipeline |
| `base_exp.yaml` | Configurações de experimentos |
| `mvp.yaml` | Configuração para versão MVP |

### Fluxo de Inferência em Batch

```
1. Coleta de Dados
   └─▶ Carregar base completa de clientes do banco de dados

2. Pré-processamento
   └─▶ Limpeza, tratamento de missing values, padronização

3. Engenharia de Features
   └─▶ Aplicar transformations (encoders, features derivadas)
   └─▶ Normalizar features numéricas (StandardScaler)

4. Inferência
   └─▶ Carregar modelo MLP treinado
   └─▶ Executar predição em lote (batch prediction)
   └─▶ Obter probabilidades de churn (0-1)

5. Post-processamento
   └─▶ Aplicar threshold (default: 0.5) para classificação
   └─▶ Ordenar por probabilidade decrescente
   └─▶ Identificar top N clientes de risco

6. Exportação
   └─▶ Salvar resultados em CSV/Parquet
   └─▶ Enviar para sistema de CRM/Marketing
   └─▶ Atualizar dashboard de monitoramento
```

## Infraestrutura de Deploy

### Containerização (Docker)

A solução é containerizada usando Docker para garantir reprodutibilidade e portabilidade:

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml src/ config/ models/ ./
RUN pip install -e .
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
```

### Orquestração (Docker Compose)

```yaml
# docker-compose.yml
services:
  mlflow:    # Tracking de experimentos
  train:     # Pipeline de treinamento
  api:       # API de inferência batch
```

### Scripts de Execução

| Script | Finalidade |
|--------|------------|
| `run_train.py` | Executa pipeline completo de treinamento |
| `run_inference.py` | Executa inferência em batch |
| `entrypoint.sh` | Script de inicialização do container |

## Monitoramento e Manutenção

### Métricas de Monitoramento

O modelo é monitorado através das seguintes métricas:

| Métrica | Descrição | Target |
|---------|-----------|--------|
| **Recall** | Capacidade de identificar churners | > 0.70 |
| **Precision** | Precisão das predições positivas | > 0.60 |
| **F1-Score** | Balance entre recall e precision | > 0.65 |
| **AUC-ROC** | Capacidade discriminativa geral | > 0.80 |

### MLflow Tracking

O MLflow é utilizado para:
- **Experiment Tracking**: Registro de todas as execuções de treino
- **Model Registry**: Versionamento de modelos em produção
- **Metrics Dashboard**: Visualização de performance ao longo do tempo

### Retreinamento

**Gatilhos para Retreinamento:**
1. **Calendar-based**: Retreinamento mensal ou trimestral
2. **Performance-based**: Queda de métricas abaixo do threshold
3. **Data Drift**: Mudança significativa na distribuição dos dados

**Pipeline de Retreinamento:**
```bash
# Executar treinamento
python run_train.py

# Validar métricas
python src/evaluation/metrics.py

# Registrar novo modelo no MLflow
mlflow models serve
```

## Considerações de Custo

### Comparação Batch vs. Streaming

| Aspecto | Batch | Streaming |
|---------|-------|-----------|
| **Infraestrutura** | Simples (CPU/GPU padrão) | Complexa (Kafka, Flink, etc.) |
| **Custo Computacional** | Baixo (processamento pontual) | Alto (24/7) |
| **Latência** | Horas/dias | Milissegundos |
| **Complexidade** | Baixa | Alta |
| **Manutenção** | Simples | Complexa |

### Estimativa de Custo (Batch Diário)

- **Processamento**: ~30 minutos em instância CPU padrão
- **Armazenamento**: Modelos + dados de treino (~500MB)
- **Custo Mensal Estimado**: 10-20x menor que solução streaming

## Conclusão

A arquitetura de deploy em batch foi selecionada por ser a abordagem mais adequada para o problema de predição de churn, considerando:

1. **Natureza do problema**: Processo lento, dependente de histórico
2. **Características do modelo**: MLP com features agregadas
3. **Requisitos de negócio**: Scoring para campanhas de retenção
4. **Restrições técnicas**: Simplicidade, custo e manutenibilidade

A solução é escalável, monitorável e pronta para produção, podendo ser executada diariamente para fornecer insights acionáveis para a equipe de marketing e retenção de clientes.

---

**Documentação Técnica Complementar:**
- [Model Card](model_card.md) - Detalhes do modelo e métricas
- [ML Canvas](ml_canvas.pdf) - Visão geral do projeto
- [API Documentation](churn_api.postman_collection.json) - Collection Postman
