
# Previsão de Carga de Energia por Região (SIN)

Este projeto implementa um pipeline completo de Machine Learning para previsão da carga média diária de energia dos subsistemas do Sistema Interligado Nacional (SIN):

- Norte
- Nordeste
- Sudeste/Centro-Oeste
- Sul

O modelo utiliza **XGBoost** e integra **dados históricos de carga do ONS** com **dados meteorológicos do INMET**, criando um sistema de previsão baseado em séries temporais e variáveis climáticas.

---

# Objetivo do Projeto

O objetivo deste projeto é desenvolver um modelo preditivo capaz de estimar a carga de energia elétrica diária por região do Brasil, utilizando:

- Dados históricos de carga
- Variáveis temporais
- Dados climáticos
- Engenharia de atributos
- Machine Learning (XGBoost)

---

# Metodologia

## Engenharia de Atributos (Feature Engineering)

O modelo utiliza as seguintes variáveis:

### Variáveis Temporais
- Mês
- Dia da semana
- Trimestre
- Ano

### Contexto de Calendário
- Feriados nacionais
- Fins de semana

### Lags e Estatísticas
- Carga do dia anterior
- Média móvel dos últimos 7 dias

### Variáveis Climáticas (INMET)
- Temperatura média diária
- Umidade relativa média diária

---

# Estratégia de Previsão – Delta de Carga

Em vez de prever diretamente a carga absoluta, o modelo prevê a variação da carga:

Delta_carga = Carga_t - Carga_{t-1}

Depois:
Carga_prevista = Carga_ontem + Delta_previsto

---

# Tecnologias Utilizadas

## Análise de Dados
- pandas
- numpy

## Machine Learning
- xgboost
- scikit-learn

## Séries Temporais
- holidays
- joblib

## Monitorização Ambiental
- codecarbon

## Visualização
- matplotlib
- seaborn

## Testes
- pytest

---

# Estrutura do Projeto

```
.
├── dataset/
├── modelos/
├── output/
├── train_pipeline.py
├── predict_2025.py
├── resultado_visualizacoes.py
├── test_logic.py
├── requirements.txt
└── README.md
```

---

# Como Executar o Projeto

## Instalar dependências
pip install -r requirements.txt

## Treinar modelos
python train_pipeline.py

## Gerar previsões
python predict_2025.py

## Gerar gráficos
python resultado_visualizacoes.py

## Executar testes
pytest test_logic.py

---
