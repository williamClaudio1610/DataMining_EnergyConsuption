# Previsao da Demanda de Energia

Projeto de Mineracao de Dados para prever a demanda de energia a partir do
dataset "household_power_consumption". O pipeline inclui EDA, preparacao
dos dados, modelos ARIMA e RNN, e modelo LSTM (PyTorch).

## Objetivo

- analisar o consumo de energia ao longo do tempo
- preparar dados horarios com features e lags
- comparar previsoes ARIMA, RNN e LSTM

## Dataset

O script de EDA espera o arquivo:

- `household_power_consumption.txt` (separador `;`, valores ausentes `?`)

## Pipeline (ordem recomendada)

1) **EDA**: `01_eda_energy.py`  
   - carrega `household_power_consumption.txt`
   - cria `DateTime` e features temporais (ano, mes, dia, hora, etc.)
   - gera graficos em `graficos_eda/`
   - salva `energy_data_processed.csv`

2) **Preparacao**: `02_data_preparation.py`  
   - trata ausentes com interpolacao + ffill/bfill
   - winsorizacao (p1 e p99)
   - agrega para frequencia horaria
   - cria lags (1h, 24h, 168h) e features temporais
   - normaliza com MinMax (0-1)
   - divide 70/15/15 (treino/validacao/teste)
   - salva:
     - `data_hourly_prepared.csv`
     - `data_train.csv`
     - `data_validation.csv`
     - `data_test.csv`
     - `scaler_minmax.pkl`
     - `graficos_eda/divisao_dados.png`

3) **ARIMA**: `03_modelo_arima.py`  
   - usa apenas `Global_active_power`
   - testa estacionaridade (ADF)
   - escolhe (p,d,q) via `auto_arima` (sem sazonalidade)
   - treina e avalia com MAE, RMSE e MAPE
   - gera graficos em `graficos_modelos/`
   - salva:
     - `arima_predictions_validation.csv`
     - `arima_predictions_test.csv`
     - `arima_model.pkl`

4) **RNN**: `04_modelo_rnn.py`  
   - usa `Global_active_power`, lags e features de tempo
   - normaliza X e y com MinMax e usa timesteps = 1
   - arquitetura: 2 camadas SimpleRNN (50) + Dropout + Dense
   - treino com EarlyStopping (MSE + Adam)
   - gera graficos em `graficos_modelos/`
   - salva:
     - `rnn_predictions_test.csv`
     - `rnn_model.h5`
     - `scaler_X_rnn.pkl`
     - `scaler_y_rnn.pkl`

5) **LSTM**: `AndreYanga_Versao/Projeto_Energia/notebooks/04_modelo_lstm.ipynb`  
   - PyTorch, janela de 24h para prever 1h
   - arquitetura: LSTM 2 camadas, hidden 50, dropout 0.2 + Linear
   - treino com MSE + Adam (lr=0.001), 50 epocas
   - avalia MAE, RMSE, MAPE e compara com ARIMA
   - arquivos referenciados no notebook:
     - entrada: `../dados/processed/energia_horario.csv`
     - arima: `../resultados/previsoes_arima.csv`
     - saida: `../modelos/lstm_modelo.pth`, `../modelos/scaler.pkl`,
       `../resultados/previsoes_lstm.csv`

## Como executar (scripts)

```bash
python 01_eda_energy.py
python 02_data_preparation.py
python 03_modelo_arima.py
python 04_modelo_rnn.py
```

Para o LSTM, abra o notebook e execute as celulas em ordem:
`AndreYanga_Versao/Projeto_Energia/notebooks/04_modelo_lstm.ipynb`.

## Principais saidas

- **EDA**: `graficos_eda/*.png`, `energy_data_processed.csv`
- **Preparacao**: `data_*`, `scaler_minmax.pkl`, `graficos_eda/divisao_dados.png`
- **ARIMA**: `graficos_modelos/*.png`, `arima_model.pkl`, `arima_predictions_*.csv`
- **RNN**: `graficos_modelos/*.png`, `rnn_model.h5`, `rnn_predictions_test.csv`
- **LSTM**: `lstm_modelo.pth`, `previsoes_lstm.csv`

## Dependencias

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `statsmodels`, `pmdarima`, `joblib`
- `tensorflow` (para o modelo RNN)
- `torch` (para o notebook LSTM)

## Estrutura relevante

- `01_eda_energy.py`
- `02_data_preparation.py`
- `03_modelo_arima.py`
- `04_modelo_rnn.py`
- `AndreYanga_Versao/Projeto_Energia/notebooks/04_modelo_lstm.ipynb`
- `graficos_eda/`, `graficos_modelos/`
