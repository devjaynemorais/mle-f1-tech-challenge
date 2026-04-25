# Model Card — Churn Prediction MLP

## Descrição do Modelo

**Nome:** MLP (Multilayer Perceptron) para Predição de Churn  
**Versão:** 1.0.0  
**Tipo:** Classificação binária (churn = 1 / não-churn = 0)  
**Framework:** PyTorch + Scikit-Learn (pré-processamento)  
**Arquitetura:** Linear(input → 64) → ReLU → Linear(64 → 1) + BCEWithLogitsLoss

---

## Uso Pretendido

**Primário:** Identificar clientes de telecom com risco de cancelamento, permitindo ações proativas de retenção.

**Usuários previstos:** Times de CRM, Marketing e Sucesso do Cliente.

**Casos fora do escopo:**
- Predição de churn em outros setores (bancário, varejo, etc.) sem retreinamento
- Decisões automatizadas sem revisão humana de alto impacto (bloqueio de conta, cobrança)

---

## Métricas de Performance

Avaliadas no conjunto de teste (30% dos dados, estratificado):

| Métrica | Valor |
|---|---|
| Recall | 0.7968 |
| Precision | 0.5422 |
| F1-Score | 0.6477 |
| AUC-ROC | 0.8519 |
| Overfitting (recall) | 3.2% |

**Early stopping:** treinamento encerrado automaticamente ao detectar ausência de melhora na val_loss (patience=10).

---

## Dados de Treinamento

**Dataset:** Telco Customer Churn — IBM (`data/raw/Telco_customer_churn.xlsx`)  
**Tamanho:** 7.043 registros  
**Features:** 20 variáveis (16 categóricas + 4 numéricas)  
**Período:** Dados históricos de clientes de uma operadora de telecom dos EUA  
**Balanceamento:** ~26% de churners — classe positiva ponderada com `pos_weight` no treinamento

---

## Limitações

1. **Distribuição geográfica:** o dataset representa clientes de uma única operadora americana. Pode não generalizar para outros mercados sem retreinamento.
2. **Deriva temporal:** o modelo não é atualizado automaticamente. Se o comportamento de churn mudar ao longo do tempo, a performance pode degradar.
3. **Variáveis ausentes:** não inclui histórico de interações com suporte, NPS ou dados de uso detalhados, que poderiam melhorar a predição.
4. **Threshold fixo:** o threshold de 0.5 não foi otimizado para o custo de negócio. Dependendo do custo de falsos negativos vs. falsos positivos, pode ser necessário ajustá-lo.
5. **CLTV como feature:** o CLTV é usado como feature de entrada, mas é em si uma estimativa. Erros no CLTV se propagam para a predição.

---

## Vieses Conhecidos

- **Gênero:** o modelo inclui `Gender` como feature. Testes de fairness não foram conduzidos formalmente. Recomenda-se avaliar paridade de performance entre grupos demográficos antes de uso em decisões de alto impacto.
- **Contrato:** clientes Month-to-month têm taxa de churn muito maior. O modelo pode superestimar risco para novos clientes nessa modalidade sem histórico suficiente.
- **Senior Citizen:** grupo minoritário no dataset; a performance pode ser menor neste segmento.

---

## Cenários de Falha

| Cenário | Efeito | Mitigação |
|---|---|---|
| Features com valores nulos na entrada | Erro na predição ou resultado incorreto | Validação Pydantic na API rejeita campos ausentes |
| Modelo desatualizado com deriva de dados | Queda de recall | Monitorar AUC mensal; retreinar se degradar >5% |
| Feature de entrada fora da distribuição de treino | Predição não confiável | Adicionar detecção de outliers no preprocessamento |
| Scaler não compatível com novo modelo | Erro de shape ou escala errada | Scaler é salvo e versionado junto com o modelo |

---

## Plano de Monitoramento

### Métricas a monitorar

| Métrica | Frequência | Alerta |
|---|---|---|
| AUC-ROC no conjunto de validação | Semanal | Degradação > 5% em relação ao baseline |
| Taxa de churn real vs. prevista | Mensal | Desvio > 10 pontos percentuais |
| Distribuição das features de entrada | Mensal | PSI > 0.2 em qualquer feature |
| Taxa de erro da API (`5xx`) | Contínuo | > 1% das requisições |
| Latência da API (`/predict`) | Contínuo | P95 > 500ms |

### Playbook de resposta

1. **Degradação de AUC > 5%:** inspecionar distribuição de features, verificar deriva → retreinar se confirmado
2. **Erro 5xx > 1%:** verificar logs de `src/api/main.py`, checar se modelo existe em `models/`
3. **Latência alta:** verificar tamanho do batch enviado, avaliar escalabilidade horizontal

---

## Informações Técnicas

- **Repositório:** POS MLE — Tech Challenge Fase 1
- **Última atualização do modelo:** 2026-04-25
- **Contato:** devmoraislacerda@gmail.com
