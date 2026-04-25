# Experimentos

Este diretório concentra todo o código de experimentação do projeto de predição de churn. O objetivo é manter a exploração e os testes de hipóteses isolados do código de produção em `src/`.

---

## Estrutura

```
experiments/
├── run_train.py             # CLI principal para rodar experimentos rastreados no MLflow
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_experimentação.ipynb
│   └── 04_modelo_mvp.ipynb
└── utils/
    ├── eda.py               # Funções auxiliares de análise exploratória
    ├── plots.py             # Visualizações reutilizáveis
    └── stats.py             # Testes estatísticos e métricas de suporte
```

Os arquivos de configuração dos experimentos ficam em `config/` na raiz do projeto:

```
config/
├── base_exp.yaml   # Baseline: todas as features opcionais desligadas
└── mvp.yaml        # Configuração do modelo MVP selecionado
```

---

## Como rodar um experimento

O script `run_train.py` é um CLI standalone. O argumento `--config` é **obrigatório** — cada experimento é descrito por um arquivo YAML.

```bash
# A partir da raiz do projeto
python experiments/run_train.py --config config/base_exp.yaml
```

> O MLflow Tracking Server precisa estar rodando antes de executar.
> Por padrão, o tracking URI aponta para `http://localhost:5000`.
>
> Para subir o servidor localmente:
> ```bash
> mlflow ui
> ```

---

## Criando um novo experimento

1. Copie o arquivo de configuração base:
   ```bash
   cp config/base_exp.yaml config/meu_experimento.yaml
   ```
2. Edite `meu_experimento.yaml`: altere `experiment.name`, ative/desative features em `features:` e ajuste hiperparâmetros em `model.params`.
3. Rode:
   ```bash
   python experiments/run_train.py --config config/meu_experimento.yaml
   ```
4. Compare os resultados na UI do MLflow (`http://localhost:5000`).

---

## O que é rastreado no MLflow

| Categoria | O que é logado |
|---|---|
| Parâmetros do modelo | `C`, `max_iter`, `class_weight`, ... |
| Features ativas | `feature__<nome>: true/false` por feature do config |
| Métricas técnicas (CV) | `recall_mean/std`, `f1_mean`, `roc_auc_mean` |
| KPIs de negócio (CV) | `captured_value_mean`, `expected_loss_mean`, `capture_value_ratio_mean`, `cltv_captured_mean` |
| Artefatos | Config YAML completo + modelo serializado (`mlflow.sklearn`) |

---

## Validação

A avaliação usa **Stratified K-Fold** (padrão: 5 folds) com **CLTV como `sample_weight`** no treino, privilegiando acertar clientes de maior valor. As métricas de negócio são calculadas por fold e reportadas como média sobre os folds.

---

## Notebooks

Os notebooks documentam a jornada de experimentação de forma incremental:

| Notebook | Conteúdo |
|---|---|
| `01_exploratory_data_analysis` | Análise exploratória dos dados brutos |
| `02_baselines` | Modelos baseline e primeiras métricas |
| `03_experimentação` | Engenharia de features e comparação de configurações |
| `04_modelo_mvp` | Consolidação do modelo selecionado para produção |
