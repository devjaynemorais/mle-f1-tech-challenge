# Telco Customer Churn - Tech Challenge FASE 1

Pipeline de Machine Learning de ponta a ponta para prever evasão de clientes (*churn*) em uma empresa de telecomunicações.

O projeto cobre EDA, modelagem com PyTorch (MLP) e Scikit-Learn, rastreamento de experimentos com MLflow, API de inferência batch com FastAPI e containerização com Docker.

---

## 🗂 Estrutura do Projeto

```
.
├── config/           # Configuração YAML (features, modelo, MLflow)
├── data/
│   ├── raw/          # Dados brutos imutáveis (Telco_customer_churn.xlsx)
│   ├── interim/      # Dados limpos intermediários
│   └── processed/    # Dados prontos para modelagem (train/test)
├── docs/             # ML Canvas, Model Card, documentação
├── models/           # Artefatos gerados pelo treino (.pt, .pkl)
├── notebooks/        # Notebooks de experimentação e análise
├── src/
│   ├── api/          # FastAPI: main.py, schemas.py, predictor.py
│   ├── data/         # Carga e limpeza dos dados brutos
│   ├── features/     # Engenharia de features e encoders
│   ├── models/       # MLP (PyTorch), treino e predição
│   ├── evaluation/   # Métricas técnicas (compute_metrics)
│   └── utils/        # Logging, EDA, plots, estatísticas
├── tests/            # Testes automatizados com pytest
├── Dockerfile        # Imagem Docker — treino, inferência e API via entrypoint.sh
├── docker-compose.yml# Orquestra mlflow + train + api
├── entrypoint.sh     # Roteador de modo: train | inference | api | mlflow
├── Makefile          # Atalhos: lint, test, train, inference, api, docker, compose
├── run_train.py      # Pipeline de treino (dados → features → modelo)
└── run_inference.py  # Pipeline de inferência (carrega modelo → prediz)
```

---

## ⚙️ Configuração (Setup)

### Pré-requisitos

- Python 3.9 ou superior
- Docker (opcional — cobre treino, inferência e API via containers, sem instalar Python localmente)

### Passo a Passo

1. Crie o ambiente virtual:

   ```bash
   python -m venv .venv
   ```

2. Ative o ambiente:

   | Sistema | Comando |
   |---|---|
   | PowerShell (Windows) | `.\.venv\Scripts\Activate.ps1` |
   | Bash (Linux/Mac) | `source .venv/bin/activate` |
   | Git Bash (Windows) | `source .venv/Scripts/activate` |

3. Instale as dependências:

   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. (Opcional) Instale o browser headless para exportação de PDF:

   ```bash
   playwright install chromium
   ```

---

## 🚀 Execução

O projeto tem **quatro fluxos independentes:**

| # | Fluxo | Quando usar | Como executar |
|---|---|---|---|
| 1 | [Experimento](#-1-experimento) | Explorar dados, features e modelos | Notebooks Jupyter |
| 2 | [Treino](#-2-treino) | Treinar o modelo de produção | `python run_train.py` |
| 3 | [Inferência](#-3-inferência) | Avaliar o modelo no conjunto de teste | `python run_inference.py` |
| 4 | [API](#-4-api-fastapi) | Servir predições batch em produção | `make api` ou Docker |

---

## 🧪 1. Experimento

> **Quando usar:** fase de exploração — análise de dados, engenharia de features, comparação de modelos e decisão do campeão.
>
> **Ferramenta:** Notebooks Jupyter em `notebooks/`

### Execute o Jupyter

```bash
jupyter notebook
```

> Notebooks com MLflow requerem o servidor em terminal separado:
> ```bash
> make mlflow
> ```

### Notebooks disponíveis

| Notebook | O que faz |
|---|---|
| `01_exploratory_data_analysis.ipynb` | EDA completa — qualidade, distribuição, correlações |
| `02_baselines.ipynb` | Baselines: DummyClassifier, Regressão Logística, MLP |
| `03_experimentação.ipynb` | Experimentação com features e hiperparâmetros + MLflow |
| `04_modelo_mvp.ipynb` | Seleção e documentação do modelo MVP final |

---

## 🏭 2. Treino

> **Quando usar:** após os experimentos, para treinar o modelo de produção com os dados completos.
>
> **Script:** `run_train.py`  
> **Config:** `config/config.yaml` — `model.name` define qual modelo treinar.  
> **Pré-requisito:** dados brutos em `data/raw/Telco_customer_churn.xlsx`.

### Passo 1 — (Opcional) Visualize os runs no MLflow UI

O treino escreve diretamente em `mlflow.db` (SQLite local) — nenhum servidor é necessário para rodar. Para visualizar os experimentos no browser:

```bash
make mlflow
# equivalente a: mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Acesse [http://127.0.0.1:5000](http://127.0.0.1:5000). Pode ser aberto antes ou depois do treino.

### Passo 2 — Execute o treino

```bash
python run_train.py
# ou: make train
```

### Passos executados em sequência

| # | Script | Entrada | Saída |
|---|---|---|---|
| 1 | `src/data/make_dataset.py` | `data/raw/Telco_customer_churn.xlsx` | `data/interim/telecom_clean.csv` |
| 2 | `src/features/build_features.py` | `data/interim/telecom_clean.csv` | `data/processed/train.csv`, `test.csv`, `models/scaler.pkl`, `models/feature_columns.json` |
| 3 | `src/models/train_model.py` | `data/processed/train.csv` | Modelo em `models/` + run no MLflow |

Se qualquer etapa falhar, o pipeline aborta e exibe o passo com erro.

### Artefatos gerados

O modelo salvo depende do `model.name` em `config/config.yaml`:

| `model.name` | Artefatos gerados |
|---|---|
| `mlp` | `models/mlp_baseline.pt` + `models/mlp_scaler.pkl` |
| `logistic_regression` | `models/logistic_regression.pkl` |
| `random_forest` | `models/rf_baseline.pkl` |
| `dummy` | `models/dummy_classifier.pkl` |

> O arquivo `models/feature_columns.json` é sempre gerado e contém a ordem exata das colunas após o one-hot encoding — necessário para a API.

### Saída no terminal (modelo atual: MLP)

```
2026-04-25 15:40:18 [INFO] __main__: Treinando modelo de produção: mlp
2026-04-25 15:40:18 [INFO] __main__: Dispositivo: cpu | max_epochs=100 | patience=10
2026-04-25 15:40:20 [INFO] __main__: Época  10/100  train_loss=0.6482  val_loss=0.6855  paciência=3/10
2026-04-25 15:40:21 [INFO] __main__: Early stopping na época 17 (melhor val_loss=0.6850)
2026-04-25 15:40:29 [INFO] __main__: Train recall=0.8234  auc=0.8721 | Test recall=0.7968  auc=0.8519 | Overfit=3.2%
```

O MLP usa **early stopping** — o treino para automaticamente quando a val_loss não melhora por `patience` épocas e restaura os melhores pesos.

---

## 🔮 3. Inferência

> **Quando usar:** com o modelo já treinado, para avaliar sua performance no conjunto de teste.
>
> **Script:** `run_inference.py`  
> **Config:** `config/config.yaml` — `model.name` define qual modelo carregar.  
> **Pré-requisito:** modelo em `models/` e dados em `data/processed/test.csv` — gerados pelo [Treino](#-2-treino).

### Execute

```bash
python run_inference.py
# ou: make inference
```

### Saída esperada (modelo atual: MLP)

```
2026-04-25 15:40:29 [INFO] __main__: Carregando modelo de produção: mlp (models/mlp_baseline.pt)
2026-04-25 15:40:30 [INFO] __main__: === Resultado — Modelo de Produção: mlp ===
2026-04-25 15:40:30 [INFO] __main__:   Recall   : 0.7968
2026-04-25 15:40:30 [INFO] __main__:   Precision: 0.5422
2026-04-25 15:40:30 [INFO] __main__:   F1       : 0.6477
2026-04-25 15:40:30 [INFO] __main__:   AUC      : 0.8519
```

Para trocar o modelo de produção, altere `model.name` e `model_path` em `config/config.yaml` e re-execute o treino.

---

## 🌐 4. API (FastAPI)

> **Quando usar:** para servir predições batch em produção. Recebe uma lista de clientes em JSON e retorna probabilidade e label de churn para cada um.
>
> **Pré-requisito:** modelo treinado em `models/` e `models/feature_columns.json` — gerados pelo [Treino](#-2-treino).

### Subir a API localmente

```bash
make api
# equivalente a: uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Acesse a documentação interativa em [http://localhost:8000/docs](http://localhost:8000/docs).

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API e o modelo estão disponíveis |
| `POST` | `/predict` | Predição batch — recebe lista de clientes, retorna probabilidades |

### Validação de entrada (Pydantic)

Toda requisição ao `/predict` é validada automaticamente pelo Pydantic v2 antes de chegar ao modelo. Erros retornam **HTTP 422** com o campo exato que falhou.

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
  "model": "mlp",
  "n_records": 1,
  "predictions": [
    {
      "churn_probability": 0.8234,
      "churn_label": 1
    }
  ]
}
```

### Rodando com Docker

> **Pré-requisito único:** [Docker Desktop](https://www.docker.com/get-started) instalado e rodando. Nenhum Python, venv ou dependência local necessária.

O `docker-compose.yml` orquestra três serviços com a mesma imagem:

| Serviço | Papel | Porta |
|---|---|---|
| `mlflow` | Tracking server — registra experimentos e métricas | 5000 |
| `train` | Pipeline de treino completo — executa e encerra | — |
| `api` | FastAPI + uvicorn — serve predições batch | 8000 |

**Passo 1 — Build (uma vez):**

```bash
docker compose build
```

**Passo 2 — Treinar:**

```bash
docker compose up -d mlflow
docker compose run --rm train
```

O `train` aguarda o MLflow estar saudável, executa `make_dataset → build_features → train_model` e encerra. Os artefatos (`.pt`, `.pkl`, `feature_columns.json`) são salvos em `./models/` no host via volume.

**Passo 3 — Subir a API:**

```bash
docker compose up -d api
```

**Testar:**

```bash
curl http://localhost:8000/health
```

| Interface | URL |
|---|---|
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://127.0.0.1:5000 |

**Parar tudo:**

```bash
docker compose down
```

---

## 🔧 Makefile

Atalhos para as operações mais comuns:

```bash
# Qualidade e testes
make lint           # ruff check .
make test           # pytest tests/ -v

# Fluxo local (requer .venv ativo)
make train          # python run_train.py
make inference      # python run_inference.py
make api            # uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
make mlflow         # mlflow server --host 127.0.0.1 --port 5000 --workers 1

# Docker Compose (fluxo completo, sem Python local)
make compose-build  # docker compose build
make compose-train  # sobe mlflow + executa treino one-shot
make compose-up     # sobe mlflow + api
make compose-down   # docker compose down
```

---

## 🧬 Testes

```bash
make test
# ou: pytest tests/ -v
```

| Arquivo | O que testa |
|---|---|
| `tests/test_smoke.py` | Carregamento do modelo, MLP, scaler e feature_columns |
| `tests/test_schema.py` | Schema dos datasets (Pandera): tipos, nulos, proporção do split |
| `tests/test_api.py` | Endpoints `/health` e `/predict`, validação Pydantic, header de latência |

### Smoke tests (`test_smoke.py`)

Verificam que a infraestrutura do modelo está intacta **sem subir a API**. Rodam em segundos e são o primeiro diagnóstico quando algo quebra após um novo treino.

| Teste | O que verifica |
|---|---|
| `test_config_carrega` | `config/config.yaml` carrega e `model.name` é um dos modelos suportados |
| `test_modelo_producao_existe` | Arquivo do modelo apontado em `config.yaml` existe no disco |
| `test_mlp_carrega_e_prediz` | MLP instancia, executa forward pass com tensor `(4, 31)` e retorna probabilidades em `[0, 1]` |
| `test_scaler_carrega` | `models/mlp_scaler.pkl` carrega e transforma uma matriz `(5, 31)` sem erros |
| `test_feature_columns_json_existe` | `models/feature_columns.json` existe, não está vazio e contém strings |

> Os smoke tests **não dependem de dados processados** — só dos artefatos gerados pelo treino (`models/`). Se falharem, o problema está nos artefatos, não na API.

---

## 📋 ML Canvas

O ML Canvas está em `docs/ml_canvas.html`, renderizado a partir de `docs/ml_canvas.json`.

**Opção 1 — Python (terminal):**
```bash
cd docs
python -m http.server 8080
```
Acesse [http://localhost:8080/ml_canvas.html](http://localhost:8080/ml_canvas.html).

**Opção 2 — VS Code:**
Clique com botão direito no `ml_canvas.html` → **"Open with Live Server"**.

**Exportar como PDF:**
```bash
python docs/export_pdf.py
```

---

## 📄 Model Card

`docs/model_card.md` documenta:
- Performance no test set (Recall, Precision, F1, AUC)
- Limitações e dados fora do escopo
- Vieses conhecidos (gênero, contrato, senior citizen)
- Cenários de falha e como mitigá-los
- Plano de monitoramento com métricas, alertas e playbook

---

## 🐛 Troubleshooting

### Venv não ativado

`run_train.py` e `run_inference.py` detectam se o venv está ativo e exibem uma mensagem clara com o comando correto antes de falhar.

### UnicodeEncodeError no Windows (MLflow emoji)

Os scripts de treino já incluem o fix automático de encoding UTF-8. Se o erro persistir em outro terminal:

```bash
# Git Bash / Bash
PYTHONIOENCODING=utf-8 python run_train.py

# PowerShell
$env:PYTHONIOENCODING='utf-8'; python run_train.py
```

---

## 📁 Dados

- **Dataset:** Telco Customer Churn — IBM (`data/raw/Telco_customer_churn.xlsx`)
- **Tamanho:** 7.043 registros, 20 features (16 categóricas + 4 numéricas)
- Os dados processados são gerados automaticamente ao executar `python run_train.py`.
