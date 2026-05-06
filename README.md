# Telco Customer Churn - Tech Challenge FASE 1

Pipeline de Machine Learning de ponta a ponta para prever evasão de clientes (*churn*) em uma empresa de telecomunicações.

O projeto cobre EDA, modelagem com Scikit-Learn (MLP + Optuna), rastreamento de experimentos com MLflow, API de inferência batch com FastAPI e containerização com Docker.

▶️ [Vídeo explicativo do projeto utilizando a metodologia STAR (Situação, Tarefa, Ação e Resultado)](https://youtu.be/KcjZENVYYs8)

---

## ⚠️ Para Rodar o Pipeline Completo

> **Leia antes de executar qualquer comando.**

O `make train` e o `make compose-train` **não treinam o modelo** — eles materializam localmente um modelo já registrado no MLflow para ser servido pela API. O treinamento completo é feito por `make setup` (via notebooks) ou por `run_experiment.py` + `run_train.py` (via scripts).

**Passos obrigatórios, em ordem:**

1. Coloque os dados brutos em `data/raw/Telco_customer_churn.xlsx`
2. `make env` — cria o ambiente virtual
3. Rode:

   ```bash
   make setup  # executa os experimentos
   ```

4. A partir daqui, escolha como servir a API:

   **Opção A — local** (usa o código e modelo do host diretamente):
   ```bash
   make api
   ```

   **Opção B — Docker** (requer rebuild para incorporar o modelo e o código atualizados):
   ```bash
   make compose-build  # builda as imagens com o config.yaml e código atualizados
   make compose-full   # sobe MLflow + API + Prometheus + Grafana
   ```

> **Por que o `run_id` fica no `config.yaml`?**
> O modelo de produção é rastreado pelo MLflow — o `run_id` aponta para o experimento exato que gerou o modelo, garantindo reprodutibilidade. Se você clonou o repositório, o `run_id` presente no `config.yaml` pertence a outro ambiente e não existe no seu MLflow local. Re-execute `make setup` para gerar e registrar o seu próprio run.

---

## ⚡ Início Rápido

> **Referência granular** — lista todos os comandos disponíveis individualmente, para quem quiser executar etapas isoladas. Para rodar o pipeline completo do zero, siga a seção [⚠️ Para Rodar o Pipeline Completo](#️-para-rodar-o-pipeline-completo) acima.

### Local

```bash
make env               # cria o ambiente virtual
make experiment        # executa pipeline de experimentação
make train             # treina MLP a partir de best_params.
make setup             # executa todos os experimentos
make lint              # linting com ruff
make test              # testes com pytest
make inference         # avalia o modelo no conjunto de teste
make api               # sobe a API em http://localhost:8000
make mlflow            # sobe MLflow UI em http://localhost:5000
```

### Docker Compose

```bash
make compose-build      # build das imagens (uma vez)
make compose-train      # sobe MLflow + materializa o modelo (requer run_id válido — veja acima)
make compose-up         # sobe MLflow + API
make compose-monitoring # sobe Prometheus + Grafana
make compose-full       # sobe tudo: MLflow + API + Prometheus + Grafana
make compose-down       # para tudo
```

### Interfaces disponíveis

| Serviço | URL | Credenciais |
|---|---|---|
| API Swagger | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## 🗂 Estrutura do Projeto

```
.
├── config/
│   ├── config.yaml        # Configuração principal (features, modelo, MLflow, produção)
│   └── best_params.yaml   # Melhores hiperparâmetros do MLP (gerado por run_experiment.py, versionado)
├── data/
│   ├── raw/               # Dados brutos imutáveis (Telco_customer_churn.xlsx)
│   ├── interim/           # Dados limpos intermediários
│   └── processed/         # Dados prontos para modelagem (train/test)
├── docs/                  # ML Canvas, Model Card, documentação
├── models/
│   └── production/        # Artefatos do modelo de produção (gerados por run_train.py)
├── notebooks/             # Notebooks de experimentação e análise
├── src/
│   ├── api/               # FastAPI: main.py, schemas.py, predictor.py
│   ├── data/              # Carga e limpeza dos dados brutos
│   ├── features/          # Engenharia de features e encoders
│   ├── models/            # Treino, produção e predição
│   ├── evaluation/        # Métricas técnicas (compute_metrics)
│   └── utils/             # Logging, EDA, plots, estatísticas
├── tests/                 # Testes automatizados com pytest
├── Dockerfile             # Imagem Docker — treino, inferência e API via entrypoint.sh
├── docker-compose.yml
├── entrypoint.sh          # Roteador de modo: train | inference | api | mlflow
├── Makefile               # Atalhos para todos os comandos do projeto
├── run_experiment.py      # Pipeline de experimentação (baseline → FE → Optuna) → best_params.yaml
├── run_train.py           # Treina MLP a partir de best_params.yaml, registra no MLflow e materializa
└── run_inference.py       # Avalia o modelo no conjunto de teste
```

---

## ⚙️ Configuração (Setup)

### Pré-requisitos

- `make` instalado localmente (Linux/Mac: nativo; Windows: via [Chocolatey](https://chocolatey.org/) com `choco install make` ou via Git Bash)
- `uv` instalado localmente
- Docker Desktop (necessário para rodar a stack via containers — MLflow, API, Prometheus, Grafana)

### Ambiente Virtual

```bash
make env   # equivalente a: uv sync --extra dev
```

Ative o venv (necessário apenas se não usar `uv run`):

| Terminal | Comando |
|---|---|
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| PowerShell (Windows) | `.\.venv\Scripts\Activate.ps1` |
| Bash / Zsh (Linux/Mac) | `source .venv/bin/activate` |

> **Sem ativar:** todos os `make` já usam `uv run` internamente — funciona de qualquer terminal sem ativação manual.

### (Opcional) Exportação de PDF

```bash
uv run playwright install chromium
```

---

## 🚀 Execução

O projeto tem **cinco fluxos independentes:**

| # | Fluxo | Quando usar | Como executar |
|---|---|---|---|
| 1 | [Experimento](#-1-experimento) | Explorar dados, features e modelos | `make experiment` ou Notebooks Jupyter |
| 2 | [Treino](#-2-treino) | Treinar e materializar o modelo de produção | `make train` ou Docker |
| 3 | [Inferência](#-3-inferência) | Avaliar o modelo no conjunto de teste | `make inference` |
| 4 | [API](#-4-api-fastapi) | Servir predições batch em produção | `make api` ou Docker |
| 5 | [Monitoramento](#-5-monitoramento-prometheus--grafana) | Observar a API em produção | Docker Compose |

---

## 🧪 1. Experimento

> **Quando usar:** fase de exploração — análise de dados, engenharia de features, comparação de modelos e decisão do campeão.

Duas formas de executar:

**Via script** (recomendado para reprodutibilidade):
```bash
make experiment
```
Executa baseline → FE rounds 1-3 → feature selection → RandomSearch → Optuna e grava `config/best_params.yaml` com os melhores hiperparâmetros do MLP.

**Via notebooks** (para análise interativa):
```bash
uv run jupyter notebook
```

> Notebooks com MLflow requerem o servidor em terminal separado:
> ```bash
> make mlflow
> ```

### Notebooks disponíveis

| Notebook | O que faz |
|---|---|
| `01_exploratory_data_analysis.ipynb` | EDA completa — qualidade, distribuição, correlações |
| `02_experimentation.ipynb` | Baselines e experimentação com features e hiperparâmetros |
| `03_modelo_mvp.ipynb` | Modelo final com Optuna + registro no MLflow + análise de fairness |

---

## 🏭 2. Treino

> **Quando usar:** para treinar o MLP de produção e materializar os artefatos para a API.

### `make train` — treino completo (recomendado)

Lê `config/best_params.yaml`, recria o split com a mesma seed, treina o MLP, otimiza o threshold por ROI, registra no MLflow e materializa o modelo. **Não depende de nenhum run anterior.**

```bash
make train
```

> **Pré-requisito:** `config/best_params.yaml` presente (já versionado no repositório com defaults). Para atualizar os hiperparâmetros com novos resultados de Optuna, execute `make experiment` antes.

### `make setup` — só materializa

Apenas baixa os artefatos de um run já registrado no MLflow para `models/production/`. Útil quando o `run_id` em `config/config.yaml` já é válido e você quer apenas servir a API.

```bash
make setup
```

### Docker

```bash
make compose-train
# equivalente a: docker compose up -d mlflow && docker compose run --rm train
```

O container `train` aguarda o MLflow estar saudável, executa `prepare_production_model.py` e encerra. Os artefatos são salvos em `./models/production/` via volume compartilhado.

---

## 🔮 3. Inferência

> **Quando usar:** com o modelo já materializado, para avaliar sua performance no conjunto de teste.
>
> **Pré-requisito:** modelo em `models/production/` — gerado pelo [Treino](#-2-treino).

```bash
make inference
# ou: uv run python run_inference.py
```

---

## 🌐 4. API (FastAPI)

> **Quando usar:** para servir predições batch em produção. Recebe uma lista de clientes em JSON e retorna probabilidade e label de churn para cada um.
>
> **Pré-requisito:** modelo materializado em `models/production/` — gerado pelo [Treino](#-2-treino).

```bash
make api
# ou: uv run uvicorn src.api.main:app --reload --host localhost --port 8000
```

Acesse a documentação interativa em [http://localhost:8000/docs](http://localhost:8000/docs).

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API e o modelo estão disponíveis |
| `POST` | `/predict` | Predição batch — recebe lista de clientes, retorna probabilidades |

### Validação de entrada (Pydantic)

Toda requisição ao `/predict` é validada automaticamente pelo Pydantic v2. Erros retornam **HTTP 422** com o campo exato que falhou.

**Campos numéricos** — rejeita valores negativos:

| Campo | Restrição |
|---|---|
| `Tenure Months` | `>= 0` |
| `Monthly Charges` | `>= 0` |
| `Total Charges` | `>= 0` |
| `CLTV` | `>= 0` |

**Campos categóricos** — apenas os valores presentes no dataset IBM Telco são aceitos (`Literal`):

| Campo | Valores aceitos |
|---|---|
| `gender` | `"Male"`, `"Female"` |
| `Senior Citizen`, `partner`, `dependents`, `Phone Service`, `Paperless Billing` | `"Yes"`, `"No"` |
| `Multiple Lines` | `"Yes"`, `"No"`, `"No phone service"` |
| `Internet Service` | `"DSL"`, `"Fiber optic"`, `"No"` |
| `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` | `"Yes"`, `"No"`, `"No internet service"` |
| `contract` | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `Payment Method` | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

> A validação categórica é importante porque o modelo foi treinado com one-hot encoding dessas categorias exatas — um valor desconhecido silenciosamente viraria coluna zero, corrompendo a predição sem nenhum erro. O Pydantic rejeita na entrada antes que isso aconteça.

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "gender": "Male",
      "Senior Citizen": "No",
      "partner": "Yes",
      "dependents": "No",
      "Tenure Months": 24,
      "Phone Service": "Yes",
      "Multiple Lines": "No",
      "Internet Service": "Fiber optic",
      "Online Security": "No",
      "Online Backup": "No",
      "Device Protection": "No",
      "Tech Support": "No",
      "Streaming TV": "Yes",
      "Streaming Movies": "Yes",
      "contract": "Month-to-month",
      "Paperless Billing": "Yes",
      "Payment Method": "Electronic check",
      "Monthly Charges": 85.0,
      "Total Charges": 2040.0,
      "CLTV": 3500
    }]
  }'
```

### Exemplo de resposta

```json
{
  "model": "MLP Optuna",
  "threshold": 0.35,
  "n_records": 1,
  "predictions": [
    {
      "churn_probability": 0.8234,
      "churn_label": 1
    }
  ]
}
```

---

## 📊 5. Monitoramento (Prometheus + Grafana)

> **Quando usar:** para observar a API em produção — latência, volume de requisições, distribuição das predições e taxa de erros em tempo real.
>
> **Pré-requisito:** Docker Desktop instalado. Datasource e dashboard já são provisionados automaticamente.

A API expõe `/metrics` no padrão Prometheus via `prometheus-fastapi-instrumentator`. O Prometheus coleta essas métricas a cada 15 s e o Grafana as visualiza em um dashboard pré-configurado.

### Métricas expostas

| Métrica | Tipo | Descrição |
|---|---|---|
| `http_request_duration_seconds` | Histogram | Latência por endpoint — base para P50, P95, P99 |
| `http_request_duration_seconds_count` | Counter | Total de requisições por endpoint e status HTTP |
| `churn_predictions_total` | Counter | Total de clientes enviados ao `/predict`, por modelo |
| `churn_probability` | Histogram | Distribuição das probabilidades retornadas pelo modelo |

### Subir o stack

```bash
make compose-full
# ou: docker compose up -d mlflow api prometheus grafana
```

### Acessar as interfaces

| Interface | URL | Credenciais |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| Métricas brutas | http://localhost:8000/metrics | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |

No Grafana: menu lateral → **Dashboards** → **"Churn API — Monitoramento"** (provisionado automaticamente).

### Dashboard — painéis disponíveis

| Painel | O que mostra |
|---|---|
| Requisições / segundo | Volume de tráfego por endpoint |
| Taxa de erros 5xx | Falhas internas da API |
| Latência P50 / P95 / P99 | Tempos de resposta reais |
| Predições (última hora) | Volume de clientes classificados |
| Distribuição de probabilidade | Concentração de scores de churn |

---

## 🧬 Testes

```bash
make test
# ou: uv run pytest tests/ -v
```

| Arquivo | O que testa |
|---|---|
| `tests/test_smoke.py` | Carregamento do modelo, MLP, scaler e feature_columns |
| `tests/test_schema.py` | Schema dos datasets (Pandera): tipos, nulos, proporção do split |
| `tests/test_api.py` | Endpoints `/health` e `/predict`, validação Pydantic, header de latência |
| `tests/test_production_model.py` | Configurações de produção, inferência e materialização do modelo |
| `tests/test_feature_transformers.py` | Transformers de features e encoders geográficos |
| `tests/test_nb03_helpers.py` | Funções auxiliares do notebook 03 (métricas, ROI, fairness) |

---

## 📋 ML Canvas e Model Card

**ML Canvas** (`docs/ml_canvas.html`):

```bash
cd docs && python -m http.server 8080
# acesse: http://localhost:8080/ml_canvas.html
```

**Model Card** (`docs/model_card.md`): documenta performance no test set, limitações, vieses conhecidos (gênero, contrato, senior citizen), cenários de falha e plano de monitoramento.

**Exportar Canvas como PDF:**
```bash
python docs/export_pdf.py
```

---

## 🐛 Troubleshooting

### `run_id` não encontrado no MLflow

O `run_id` em `config/config.yaml` pertence a outro ambiente. Execute `make setup` para gerar e registrar o seu próprio run automaticamente.

### Venv não ativado

`run_train.py` e `run_inference.py` detectam se o venv está ativo e exibem uma mensagem clara com o comando correto. Para evitar, execute com `uv run`.

### UnicodeEncodeError no Windows

```bash
# Git Bash
PYTHONIOENCODING=utf-8 uv run python run_train.py

# PowerShell
$env:PYTHONIOENCODING='utf-8'; uv run python run_train.py
```

---

## 📁 Dados

- **Dataset:** Telco Customer Churn — IBM (`data/raw/Telco_customer_churn.xlsx`)
- **Tamanho:** 7.043 registros, 20 features (16 categóricas + 4 numéricas)
