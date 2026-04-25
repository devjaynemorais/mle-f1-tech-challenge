# Telco Customer Churn - Tech Challenge FASE 1

Este projeto tem como objetivo construir um pipeline de Machine Learning de ponta a ponta para prever a evasão de clientes (*churn*) em uma empresa de telecomunicações.

O desenvolvimento abrange desde a Análise Exploratória de Dados (EDA) até o treinamento e produtização do modelo, seguindo boas práticas de MLOps. Durante a modelagem, as métricas e experimentos das Redes Neurais (PyTorch) e dos baselines (Scikit-Learn) são monitorados com MLflow.

## 🗂 Estrutura do Projeto

- `config/`: Arquivos YAML de configuração dos experimentos (features, modelo, validação, MLflow).
- `data/`: Diretório do Data Lake Local.
  - `raw/`: Dados originais e brutos, imutáveis (ex: `Telco_customer_churn.xlsx`).
  - `interim/`: Dados intermediários em transformação.
  - `processed/`: Dados finais, limpos e prontos para modelagem.
  - `external/`: Dados de fontes de terceiros.
- `docs/`: Documentações de regras de negócios, arquitetura e anotações.
- `models/`: Artefatos serializados gerados pelo pipeline (pesos `.pth`, modelos `.pkl`, scalers).
- `notebooks/`: Notebooks Jupyter de experimentação e análise (EDA, baselines, MLP, MVP).
- `src/`: Pacotes Python do projeto, organizados por responsabilidade:
  - `src/data/`: Carga, limpeza e divisão dos dados brutos.
  - `src/features/`: Engenharia de features e encoders customizados.
  - `src/models/`: Arquitetura MLP (PyTorch), treino de produção e inferência.
  - `src/evaluation/`: Métricas ML (`compute_metrics`) e métricas de negócio (CLTV, perda esperada).
  - `src/utils/`: Funções auxiliares de EDA, plots e estatísticas (usadas nos notebooks).
- `tests/`: Suítes de testes automatizados com `pytest`.

## ⚙️ Configuração (Setup)

Utilizamos o `pyproject.toml` como a nossa **Single Source of Truth** (única fonte da verdade) tanto para o empacotamento do projeto quanto para dependências de desenvolvedores.

### Pré-requisitos
- Python 3.9 ou superior instalado.

### Passo a Passo

1. Crie o ambiente virtual na pasta do projeto:
   ```bash
   python -m venv .venv
   ```

2. Ative o ambiente virtual:
   - **No PowerShell (Windows)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **No Bash (Linux/Mac)**:
     ```bash
     source .venv/bin/activate
     ```
   - **No Git Bash (Windows)**:
     ```bash
     source .venv/Scripts/activate
     ```

3. Atualize seu `pip` e instale as dependências (com o pacote em modo editável):
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. Instale o browser headless para exportação de PDF:
   ```bash
   playwright install chromium
   ```

## 🚀 Execução

Ative o ambiente virtual antes de qualquer execução:

| Sistema | Comando |
|---|---|
| PowerShell (Windows) | `.\.venv\Scripts\Activate.ps1` |
| Bash (Linux/Mac) | `source .venv/bin/activate` |
| Git Bash (Windows) | `source .venv/Scripts/activate` |

O projeto tem **três fluxos independentes:**

| # | Fluxo | Quando usar | Como executar |
|---|---|---|---|
| 1 | [Experimento](#-1-experimento) | Explorar e validar hipóteses | Notebooks Jupyter |
| 2 | [Treino](#-2-treino) | Treinar os modelos para produção | `python run_train.py` |
| 3 | [Inferência](#-3-inferência) | Avaliar modelos treinados | `python run_inference.py` |

---

## 🧪 1. Experimento

> **Quando usar:** fase de exploração — analisar dados, testar features, comparar modelos e decidir qual configuração seguir para produção.
>
> **Ferramenta:** Notebooks Jupyter em `notebooks/`  
> **Pré-requisito:** ambiente virtual ativo e dependências instaladas.

### Execute o Jupyter

```bash
jupyter notebook
```

### Notebooks disponíveis

| Notebook | O que faz |
|---|---|
| `01_exploratory_data_analysis.ipynb` | EDA completa — qualidade, distribuição, correlações |
| `02_baselines.ipynb` | Modelos baseline: DummyClassifier, Regressão Logística, MLP |
| `03_experimentação.ipynb` | Experimentação com features e hiperparâmetros, tracking MLflow |
| `04_modelo_mvp.ipynb` | Seleção e documentação do modelo MVP final |

> Notebooks que usam MLflow requerem o servidor em terminal separado:
> ```bash
> mlflow server --host 127.0.0.1 --port 5000
> ```

---

## 🏭 2. Treino

> **Quando usar:** após os experimentos, para treinar o modelo de produção com os dados completos, gerar os artefatos e registrar no MLflow.
>
> **Script:** `run_train.py`  
> **Config:** `config/config.yaml` — `model.name` define qual modelo treinar.  
> **Pré-requisito:** dados brutos em `data/raw/Telco_customer_churn.xlsx`.

### Passo 1 — (Opcional) Suba o MLflow para visualizar os runs

```bash
mlflow server --host 127.0.0.1 --port 5000
```

O pipeline roda sem o servidor; os runs ficam salvos localmente em `mlruns/`.

### Passo 2 — Execute o treino

```bash
python run_train.py
```

### Passos executados em sequência

| # | Script | Entrada | Saída |
|---|---|---|---|
| 1 | `src/data/make_dataset.py` | `data/raw/Telco_customer_churn.xlsx` | `data/interim/telecom_clean.csv` |
| 2 | `src/features/build_features.py` | `data/interim/telecom_clean.csv` | `data/processed/train.csv`, `test.csv`, `models/scaler.pkl` |
| 3 | `src/models/train_model.py` | `data/processed/train.csv` | Modelo em `models/` + run no MLflow |

Se qualquer etapa falhar, o pipeline aborta e exibe o passo com erro.

### Artefatos gerados

O modelo salvo depende do `model.name` em `config/config.yaml`:

| `model.name` | Artefato salvo em `model_path` |
|---|---|
| `logistic_regression` | `models/logistic_regression.pkl` |
| `random_forest` | `models/rf_baseline.pkl` |
| `mlp` | `models/mlp_baseline.pt` + `models/mlp_scaler.pkl` |
| `dummy` | `models/dummy_classifier.pkl` |

### Saída no terminal

```
--- Treinando modelo de produção: logistic_regression ---
  CV recall=0.8149  CV auc=0.8572
  Train — recall=0.8211  auc=0.8623
  Test  — recall=0.7950  auc=0.8531
  Overfitting recall: 3.2%

train_model.py concluído! Modelo salvo em models/logistic_regression.pkl
```

---

## 🔮 3. Inferência

> **Quando usar:** com o modelo já treinado, para avaliar sua performance no conjunto de teste.
>
> **Script:** `run_inference.py`  
> **Config:** `config/config.yaml` — `model.name` define qual modelo carregar.  
> **Pré-requisito:** modelo treinado em `models/` e dados em `data/processed/test.csv` — gerados pelo [Treino](#-2-treino).

### Execute a inferência

```bash
python run_inference.py
```

### Saída esperada

```
=== Resultado — Modelo de Produção: logistic_regression ===
  Recall   : 0.7950
  Precision: 0.5168
  F1       : 0.6264
  AUC      : 0.8531
```

Para trocar o modelo avaliado, altere `model.name` e `model_path` em `config/config.yaml` e re-execute o treino.

---

## 🐛 Troubleshooting

### UnicodeEncodeError ao rodar no Windows

Scripts que usam MLflow (`experiments/run_train.py`, `src/models/train_model.py`) podem falhar no Windows com o erro abaixo ao finalizar um run:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3c3'
```

**Causa:** o MLflow imprime um emoji ao encerrar cada run, e o console Windows usa cp1252 por padrão, que não suporta esse caractere.

**Solução aplicada:** os scripts já incluem o fix automático de encoding para Windows. Caso o erro persista (ex: em outro terminal), use:

```bash
# Git Bash / Bash
PYTHONIOENCODING=utf-8 python run_train.py

# PowerShell
$env:PYTHONIOENCODING='utf-8'; python run_train.py
```

---

## 🔧 Outros comandos úteis

- **Testes unitários**:
  ```bash
  pytest
  ```

- **Linting**:
  ```bash
  ruff check .
  ```

- **Notebooks** (requer MLflow rodando):
  ```bash
  mlflow server --host 127.0.0.1 --port 5000  # terminal 1
  jupyter notebook                              # terminal 2
  ```

## 📓 Notebooks

Os notebooks seguem uma ordem progressiva de análise e experimentação:

| Notebook | Descrição |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Análise exploratória dos dados (EDA) |
| `02_baselines.ipynb` | Modelos baseline (Dummy, Regressão Logística, MLP) |
| `03_experimentação.ipynb` | Experimentação e refinamento de modelos |
| `04_modelo_mvp.ipynb` | Modelo final MVP |

Para abrir os notebooks:
```bash
jupyter notebook
```

## 📋 ML Canvas

O ML Canvas do projeto está em `docs/ml_canvas.html` e é renderizado a partir dos dados em `docs/ml_canvas.json`.

Para editar o conteúdo, modifique o `ml_canvas.json` e recarregue a página.

**Opção 1 — Python (terminal):**
```bash
cd docs
python -m http.server 8080
```
Acesse [http://localhost:8080/ml_canvas.html](http://localhost:8080/ml_canvas.html) no navegador.

**Opção 2 — VS Code:**
Instale a extensão [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer), clique com botão direito no `ml_canvas.html` → **"Open with Live Server"**.
> Vantagem: recarrega automaticamente ao salvar o `ml_canvas.json`.

**Exportar como PDF:**
```bash
python docs/export_pdf.py
```
Gera `docs/ml_canvas.pdf` em formato A3 paisagem com fidelidade total ao visual do HTML.

## 📁 Dados

O dataset utilizado é o **Telco Customer Churn** (`data/raw/Telco_customer_churn.xlsx`).  
Os dados processados são gerados automaticamente na pasta `data/processed/` ao executar `python run_train.py`.