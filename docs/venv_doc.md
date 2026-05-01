# Estratégia para o Ambiente Virtual (`.venv`)

## Objetivo

Documentar a fragilidade observada no ambiente virtual atual do projeto e propor uma estratégia mais estável para setup e execução local.

## Situação Atual

O ambiente virtual atual foi criado com `uv`, provavelmente a partir de um fluxo semelhante a:

```powershell
uv sync
```

Ao inspecionar o `pyvenv.cfg`, foi identificado que a `.venv` está associada ao runtime:

```text
C:\Users\jooar\AppData\Local\Python\pythoncore-3.14-64
```

Também foi identificado que o ambiente está usando:

- Python `3.14.3`
- runtime gerenciado fora do diretório do projeto
- dependência implícita do Python resolvido pelo `uv`

## Fragilidade Observada

O principal problema não é apenas "usar `uv`", e sim a combinação abaixo:

1. `.venv` acoplada a um runtime fora do projeto
2. uso de Python `3.14`, que ainda pode trazer instabilidade de compatibilidade no ecossistema
3. fluxo híbrido entre ativação manual de venv e gerenciamento por `uv`

Essa combinação torna o setup mais frágil porque o projeto passa a depender de um interpretador base que:

- não está explicitamente padronizado no repositório;
- pode variar entre máquinas;
- pode mudar de comportamento quando o runtime gerenciado pelo `uv` muda;
- pode gerar falhas difíceis de diagnosticar quando ferramentas como `pytest`, `python` e scripts do projeto deixam de iniciar corretamente.

## Por que isso é um risco neste projeto

Este repositório depende de bibliotecas como:

- `torch`
- `mlflow`
- `pandera`
- `scikit-learn`

Essas bibliotecas tendem a funcionar com maior previsibilidade em versões já bastante consolidadas do Python, especialmente no intervalo `3.10` a `3.12`. Ao usar `3.14`, o risco não é necessariamente falha imediata, mas sim:

- incompatibilidade parcial com dependências;
- comportamento inconsistente entre ambientes;
- mais atrito no onboarding;
- maior custo de manutenção para um problema que não traz ganho direto ao experimento.

## Solução Proposta

Padronizar o ambiente do projeto com:

- `uv` como gerenciador de ambiente e dependências
- Python `3.12` como versão alvo
- recriação limpa da `.venv`
- uso preferencial de `uv run` para execução dos comandos

## Por que adotar essa solução

### 1. Reduz acoplamento implícito

Ao fixar a versão do Python e recriar a `.venv`, o projeto deixa de depender de um estado anterior pouco transparente do runtime.

### 2. Melhora a previsibilidade

Python `3.12` oferece um ponto mais estável para o stack atual, com menor risco de incompatibilidade em bibliotecas de ML e infraestrutura.

### 3. Simplifica o uso diário

Ao usar `uv run`, reduzimos a dependência de ativação manual do ambiente no shell. Isso evita uma classe comum de erros em que o usuário acredita estar no ambiente correto, mas está executando outro Python.

### 4. Facilita onboarding e manutenção

Um fluxo de setup padronizado ajuda qualquer pessoa do time a reproduzir o ambiente com menos ambiguidade.

## Fluxo Recomendado

### Recriação do ambiente

```powershell
Remove-Item -Recurse -Force .venv
uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev
```

### Execução recomendada

Em vez de depender apenas de:

```powershell
python run_train.py
pytest
```

preferir:

```powershell
uv run python run_train.py
uv run python run_inference.py
uv run pytest
```

## Diretriz para o projeto

Seguindo esta proposta, a convenção recomendada para o repositório passa a ser:

- manter `uv` como ferramenta principal;
- evitar depender de uma `.venv` antiga criada com outra versão de Python;
- padronizar Python `3.12` como base do ambiente local;
- privilegiar `uv run` para treino, inferência e testes.

## Conclusão

A fragilidade atual não vem de uma falha conceitual do `uv`, mas de um ambiente que ficou implicitamente dependente de um runtime externo em Python `3.14`, sem uma convenção suficientemente explícita no projeto.

A solução proposta preserva o uso de `uv`, mas torna o setup mais robusto, previsível e fácil de reproduzir. Para este projeto, essa é uma escolha mais segura do que manter a `.venv` atual ou seguir operando com uma versão de Python ainda pouco estabilizada para o stack adotado.
