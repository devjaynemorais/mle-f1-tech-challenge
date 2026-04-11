# Telco Customer Churn - Tech Challenge FASE 1

Este projeto tem como objetivo construir um pipeline de Machine Learning de ponta a ponta para prever a evasão de clientes (*churn*) em uma empresa de telecomunicações.

O desenvolvimento abrange desde a Análise Exploratória de Dados (EDA) até o treinamento e produtização do modelo, seguindo boas práticas de MLOps. Durante a modelagem, as métricas e experimentos das Redes Neurais (PyTorch) e dos baselines (Scikit-Learn) são monitorados com MLflow.

## 🗂 Estrutura do Projeto

- `data/`: Diretório do Data Lake Local.
  - `raw/`: Dados originais e brutos, imutáveis (ex: `Telco-Customer-Churn.csv`).
  - `interim/`: Dados intermediários em transformação.
  - `processed/`: Dados finais, limpos e prontos para modelagem.
  - `external/`: Dados de fontes de terceiros.
- `docs/`: Documentações de regras de negócios, arquitetura e anotações.
- `models/`: Artefatos serializados (pesos de modelo, encodings, etc).
- `notebooks/`: Notebooks Jupyter para experimentação e análises de dados rápidas.
- `src/`: Código e pipelines de pré-processamento, modelo, avaliação e a API final.
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

## 🚀 Execução 

Com o ambiente ativado e dependências resolvidas, você já pode operar todas as faces analíticas do projeto.

- **Exploração e Notebooks**:
  ```bash
  jupyter notebook
  ```
  *(Confira o notebook inicial: `notebooks/01_exploratory_data_analysis.ipynb`)*

- **Executar a suíte de Testes Unitários**:
  ```bash
  pytest
  ```

- **Verificação de Padrão e Linting**:
  Mantemos o formato e a limpeza de código com ferramentas baseadas nas configurações globais do `pyproject.toml`.
  ```bash
  black .
  isort .
  ruff check .
  ```

- **Executar o Pipeline Completo** (pré-processamento → features → treino → predição):
  ```bash
  python run_pipeline.py
  ```
  O pipeline executa automaticamente na seguinte ordem:
  1. `src/data/make_dataset.py` — carrega e limpa os dados brutos
  2. `src/features/build_features.py` — engenharia de features
  3. `src/models/train_model.py` — treina o modelo e registra experimento no MLflow
  4. `src/models/predict_model.py` — gera predições

- **Visualizar experimentos no MLflow UI**:
  ```bash
  mlflow ui
  ```
  Acesse [http://localhost:5000](http://localhost:5000) no navegador para ver métricas, parâmetros e artefatos dos experimentos registrados.

## 📓 Notebooks

Os notebooks seguem uma ordem progressiva de análise e experimentação:

| Notebook | Descrição |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Análise exploratória dos dados (EDA) |
| `02_baseline.ipynb` | Modelos baseline (Dummy, Regressão Logística, MLP) |
| `03_experimentação.ipynb` | Experimentação e refinamento de modelos |
| `04_modelo_mvp.ipynb` | Modelo final MVP |

Para abrir os notebooks:
```bash
jupyter notebook
```

## 📁 Dados

O dataset utilizado é o **Telco Customer Churn** (`data/raw/Telco-Customer-Churn.csv`).  
Os dados processados são gerados automaticamente na pasta `data/processed/` ao executar o pipeline.