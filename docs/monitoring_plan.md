# Plano de Monitoramento - API de Churn Prediction

## Visão Geral

Este documento descreve o plano completo de monitoramento para a API de predição de churn, incluindo métricas, alertas e procedimentos de resposta a incidentes.

## Arquitetura de Monitoramento

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   API       │────▶│  Prometheus │────▶│   Grafana   │
│  FastAPI    │     │   :9090     │     │   :3000     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    │
       │                    ▼                    │
       │            ┌─────────────┐              │
       │            │   Alertas   │──────────────┘
       │            │  (Alertmgr) │
       ▼            └─────────────┘
┌─────────────┐
│   MLflow    │
│   :5000     │
└─────────────┘
```

## Métricas Coletadas

### 1. Métricas de Negócio/ML

| Métrica | Tipo | Descrição | Labels |
|---------|------|-----------|--------|
| `churn_predictions_total` | Counter | Total de predições realizadas | `model` |
| `churn_probability` | Histogram | Distribuição das probabilidades de churn | - |
| `churn_prediction_confidence` | Gauge | Média da confiança das últimas predições | `model` |

### 2. Métricas de Performance

| Métrica | Tipo | Descrição | Labels |
|---------|------|-----------|--------|
| `churn_api_request_latency_seconds` | Histogram | Latência das requisições da API | `endpoint` |
| `churn_prediction_latency_seconds` | Histogram | Tempo de predição do modelo | `model` |

### 3. Métricas de Erros

| Métrica | Tipo | Descrição | Labels |
|---------|------|-----------|--------|
| `churn_api_requests_total` | Counter | Total de requisições | `endpoint`, `method`, `status` |
| `churn_api_request_errors_total` | Counter | Total de erros na API | `endpoint`, `error_type` |
| `churn_prediction_errors_total` | Counter | Erros durante predições | `model`, `error_type` |

### 4. Métricas de Saúde do Modelo

| Métrica | Tipo | Descrição | Labels |
|---------|------|-----------|--------|
| `churn_model_loaded` | Gauge | Modelo carregado (1) ou não (0) | `model_name` |
| `churn_model_load_time_seconds` | Gauge | Tempo de carregamento do modelo | `model_name` |

## Alertas Configurados

### Alertas Críticos (Severity: critical)

| Alerta | Condição | Duração | Ação |
|--------|----------|---------|------|
| `APIDown` | API não responde | 1 min | Acionar equipe imediatamente |
| `APIHealthyButModelDown` | API no ar, modelo indisponível | 2 min | Verificar carregamento do modelo |
| `ScrapeTargetsDown` | Serviços não respondem ao Prometheus | 2 min | Verificar infraestrutura |

### Alertas Altos (Severity: high)

| Alerta | Condição | Duração | Ação |
|--------|----------|---------|------|
| `HighErrorRate` | Taxa de erro > 5% | 2 min | Investigar logs de erro |
| `ModelNotLoaded` | Modelo não carregado | 1 min | Recarregar modelo |
| `PredictionErrors` | Erros durante predição | 2 min | Verificar dados de entrada |
| `GrafanaDown` | Grafana indisponível | 2 min | Reiniciar serviço |
| `MLflowDown` | MLflow indisponível | 3 min | Verificar MLflow |

### Alertas Médios (Severity: medium)

| Alerta | Condição | Duração | Ação |
|--------|----------|---------|------|
| `HighLatency` | Latência p95 > 1s | 5 min | Otimizar consultas/modelo |
| `HighPredictionLatency` | Latência predição p95 > 500ms | 5 min | Otimizar inferência |

### Alertas de Aviso (Severity: warning)

| Alerta | Condição | Duração | Ação |
|--------|----------|---------|------|
| `LowPredictionConfidence` | Confiança média < 30% | 10 min | Avaliar qualidade do modelo |
| `HighPredictionVolume` | Volume > 10 pred/segundo | 5 min | Planejar escalonamento |
| `SlowModelLoad` | Carregamento > 30s | 1 min | Otimizar carregamento |
| `PrometheusRestart` | Prometheus reiniciou | 1 min | Verificar estabilidade |

## Playbook de Resposta a Incidentes

### 1. API Fora do Ar (APIDown)

**Sintomas:**
- Endpoint `/health` não responde
- Alerta `APIDown` disparado

**Ações:**
1. Verificar status do container: `docker compose ps`
2. Checar logs: `docker compose logs api`
3. Reiniciar serviço: `docker compose restart api`
4. Se persistir, verificar:
   - Uso de memória/CPU
   - Espaço em disco
   - Conexão com banco de dados

### 2. Modelo Não Carregado (ModelNotLoaded / APIHealthyButModelDown)

**Sintomas:**
- API responde mas não faz predições
- Métrica `churn_model_loaded` = 0

**Ações:**
1. Verificar logs de carregamento: `docker compose logs api \| grep "carregado"`
2. Checar se arquivo do modelo existe
3. Verificar URI do modelo no MLflow
4. Recarregar modelo reiniciando a API
5. Se persistir, verificar:
   - Compatibilidade de versão do modelo
   - Dependências do MLflow

### 3. Alta Taxa de Erros (HighErrorRate)

**Sintomas:**
- Taxa de erro > 5%
- Alerta `HighErrorRate` disparado

**Ações:**
1. Identificar tipo de erro: `churn_api_request_errors_total`
2. Verificar logs de erro: `docker compose logs api \| grep ERROR`
3. Analisar padrões:
   - Erros de validação de dados?
   - Erros de timeout?
   - Erros de modelo?
4. Ações corretivas:
   - Corrigir schema de validação
   - Aumentar timeout
   - Atualizar modelo

### 4. Latência Alta (HighLatency / HighPredictionLatency)

**Sintomas:**
- Latência p95 > 1s (API) ou > 500ms (predição)
- Alertas correspondentes disparados

**Ações:**
1. Verificar métricas de latência por endpoint
2. Analisar logs de requisições lentas
3. Verificar:
   - Uso de CPU/memória
   - Tamanho do batch de predição
   - Complexidade do modelo
4. Otimizações possíveis:
   - Reduzir tamanho do batch
   - Usar modelo mais leve
   - Implementar cache

### 5. Baixa Confiança nas Predições (LowPredictionConfidence)

**Sintomas:**
- Confiança média < 30%
- Alerta `LowPredictionConfidence` disparado

**Ações:**
1. Analisar distribuição de probabilidades
2. Verificar se dados de entrada estão no domínio esperado
3. Avaliar necessidade de:
   - Retreinar modelo
   - Ajustar threshold
   - Coletar mais dados

### 6. Erros de Predição (PredictionErrors)

**Sintomas:**
- Erros durante execução de predições
- Alerta `PredictionErrors` disparado

**Ações:**
1. Identificar tipo de erro (`error_type` label)
2. Verificar dados de entrada:
   - Valores nulos?
   - Tipos incorretos?
   - Features faltando?
3. Validar schema de entrada
4. Se erro persistente, considerar:
   - Atualizar pré-processamento
   - Retreinar modelo com mais dados

## Dashboards Grafana

### Dashboard Principal - Visão Geral

**Painéis:**
1. **Status da API** - Saúde geral (up/down)
2. **Requisições por Segundo** - Volume de tráfego
3. **Taxa de Erros** - Percentual de erros
4. **Latência (p50, p95, p99)** - Performance da API
5. **Predições Realizadas** - Total acumulado
6. **Confiança Média** - Qualidade das predições

### Dashboard de Modelo ML

**Painéis:**
1. **Status do Modelo** - Carregado/Não carregado
2. **Tempo de Carregamento** - Performance do load
3. **Latência de Predição** - Tempo de inferência
4. **Distribuição de Probabilidades** - Histograma de churn
5. **Erros por Tipo** - Breakdown de erros

### Dashboard de Infraestrutura

**Painéis:**
1. **Status dos Serviços** - Todos os containers
2. **Uso de Recursos** - CPU/Memória
3. **Disponibilidade** - Uptime dos serviços

## Procedimentos de Manutenção

### Diário
- [ ] Verificar dashboards no Grafana
- [ ] Checar alertas pendentes
- [ ] Revisar logs de erros críticos

### Semanal
- [ ] Analisar tendências de latência
- [ ] Revisar volume de predições
- [ ] Avaliar qualidade das predições (confiança)

### Mensal
- [ ] Revisar thresholds de alertas
- [ ] Analisar necessidade de retreinamento
- [ ] Planejar melhorias de performance

## Contatos de Emergência

| Função | Responsável | Contato |
|--------|-------------|---------|
| On-call Primary | Engenharia de ML | slack/#mle-oncall |
| On-call Secondary | DevOps | slack/#devops |
| Tech Lead | Engenharia | slack/#mle-tech |

## Referências

- [Documentação Prometheus](https://prometheus.io/docs/)
- [Documentação Grafana](https://grafana.com/docs/)
- [Runbooks Internos](./runbooks/)