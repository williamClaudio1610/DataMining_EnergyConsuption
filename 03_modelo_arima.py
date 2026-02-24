import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODELO ARIMA - PREVISÃO DE DEMANDA DE ENERGIA")
print("="*70)

# Carregar dados
print("\n[1] Carregando dados...")
train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)
val = pd.read_csv('data_validation.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)

# ============================================================
# DADOS DE ENTRADA: Apenas Global_active_power
# Motivo: ARIMA trabalha melhor com uma única variável
# ============================================================
target = 'Global_active_power'
train_series = train[target]
val_series = val[target]
test_series = test[target]

print(f"Treino: {len(train_series)} registros")
print(f"Validação: {len(val_series)} registros")
print(f"Teste: {len(test_series)} registros")

# [2] VERIFICAR ESTACIONARIDADE (teste ADF)
print("\n[2] Testando estacionaridade (ADF)...")
adf_result = adfuller(train_series)
print(f"Estatística ADF: {adf_result[0]:.4f}")
print(f"P-value: {adf_result[1]:.6f}")
if adf_result[1] < 0.05:
    print("→ Série é estacionária (p < 0.05)")
else:
    print("→ Série NÃO é estacionária (p >= 0.05)")

# ============================================================
# PARÂMETROS (p,d,q): AUTOMÁTICO com auto_arima
# Motivo: Testa múltiplas combinações e escolhe automaticamente
#         com base no critério AIC (mais eficiente)
# DECISÃO: ARIMA simples (sem sazonalidade)
# Justificação:
#   - SARIMA(24h) mostrou-se computacionalmente inviável
#   - Features de lag já capturam padrões temporais
#   - Melhor trade-off entre precisão e tempo de execução
# ============================================================
print("\n[3] Determinando melhores parâmetros automaticamente...")
print("Usando auto_arima (pode demorar 5-10 minutos)...")

auto_model = auto_arima(
    train_series,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,  # Determina automaticamente
    seasonal=False,  # SEM sazonalidade
    stepwise=True,  # Busca stepwise (mais rápida)
    suppress_warnings=True,
    error_action='ignore',
    trace=True  # Mostra progresso
)

best_order = auto_model.order
print(f"\nMelhores parâmetros encontrados: (p, d, q) = {best_order}")
print(f"AIC: {auto_model.aic():.2f}")

# [4] TREINAR MODELO FINAL
print("\n[4] Treinando modelo ARIMA final...")
final_model = ARIMA(train_series, order=best_order)
final_result = final_model.fit()
print("Modelo treinado!")
print(final_result.summary())

# ============================================================
# JANELA DE PREVISÃO: 24 horas
# Motivo: Alinha com o objetivo do projeto de prever
#         a demanda em diferentes períodos do dia
# ============================================================

# [5] PREVISÃO NO CONJUNTO DE VALIDAÇÃO
print("\n[5] Fazendo previsões no conjunto de validação...")
val_predictions = final_result.forecast(steps=len(val_series))

# ============================================================
# MÉTRICAS: MAE + RMSE
# MAE: Erro médio absoluto (mais intuitivo)
# RMSE: Raiz do erro médio quadrático (penaliza erros grandes)
# ============================================================
val_mae = mean_absolute_error(val_series, val_predictions)
val_rmse = np.sqrt(np.mean((val_series.values - val_predictions)**2))


print(f"\n--- Resultados Validação ---")
print(f"MAE:  {val_mae:.4f} kW")
print(f"RMSE: {val_rmse:.4f} kW")
print(f"MAPE: {np.mean(np.abs((val_series.values - val_predictions) / val_series.values)) * 100:.2f}%")

# [6] RETREINAR COM TREINO + VALIDAÇÃO PARA O TESTE
print("\n[6] Retreinando modelo com treino + validação...")
train_val_series = pd.concat([train_series, val_series])
final_model_test = ARIMA(train_val_series, order=best_order)
final_result_test = final_model_test.fit()

print("\n[7] Fazendo previsões no conjunto de teste...")
test_predictions = final_result_test.forecast(steps=len(test_series))

# Métricas do teste
test_mae = mean_absolute_error(test_series, test_predictions)
test_rmse = np.sqrt(np.mean((test_series.values - test_predictions)**2))
test_mape = np.mean(np.abs((test_series.values - test_predictions) / test_series.values)) * 100

print(f"\n--- Resultados Teste ---")
print(f"MAE:  {test_mae:.4f} kW")
print(f"RMSE: {test_rmse:.4f} kW")
print(f"MAPE: {test_mape:.2f}%")

# [8] VISUALIZAÇÕES
print("\n[8] Gerando gráficos...")

import os
if not os.path.exists('graficos_modelos'):
    os.makedirs('graficos_modelos')

# Gráfico 1: Validação — real vs previsão (primeiras 500 horas)
fig, ax = plt.subplots(figsize=(15, 6))
plot_range = min(500, len(val_series))
ax.plot(val_series.index[:plot_range], val_series.values[:plot_range], 
        label='Real', color='blue', linewidth=2)
ax.plot(val_series.index[:plot_range], val_predictions[:plot_range], 
        label='Previsão ARIMA', color='red', linewidth=2, alpha=0.7)
ax.set_title('ARIMA — Validação: Real vs Previsão (primeiras 500h)', fontweight='bold', fontsize=14)
ax.set_xlabel('Data')
ax.set_ylabel('Potência Ativa (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/arima_validacao.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 2: Teste — real vs previsão (primeiras 500 horas)
fig, ax = plt.subplots(figsize=(15, 6))
plot_range = min(500, len(test_series))
ax.plot(test_series.index[:plot_range], test_series.values[:plot_range], 
        label='Real', color='blue', linewidth=2)
ax.plot(test_series.index[:plot_range], test_predictions[:plot_range], 
        label='Previsão ARIMA', color='red', linewidth=2, alpha=0.7)
ax.set_title('ARIMA — Teste: Real vs Previsão (primeiras 500h)', fontweight='bold', fontsize=14)
ax.set_xlabel('Data')
ax.set_ylabel('Potência Ativa (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/arima_teste.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 3: Erro ao longo do tempo (teste - primeiras 500h)
errors = test_series.values - test_predictions
fig, ax = plt.subplots(figsize=(15, 5))
plot_range = min(500, len(test_series))
ax.plot(test_series.index[:plot_range], errors[:plot_range], color='orange', linewidth=1)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(test_series.index[:plot_range], errors[:plot_range], 0, alpha=0.3, color='orange')
ax.set_title('ARIMA — Erro ao Longo do Tempo (primeiras 500h)', fontweight='bold', fontsize=14)
ax.set_xlabel('Data')
ax.set_ylabel('Erro (kW)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/arima_erro.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfico 4: Distribuição dos erros
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Distribuição dos Erros', fontweight='bold')
axes[0].set_xlabel('Erro (kW)')
axes[0].set_ylabel('Frequência')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(test_predictions, test_series.values, alpha=0.5, s=10)
axes[1].plot([test_series.min(), test_series.max()], 
             [test_series.min(), test_series.max()], 
             'r--', linewidth=2, label='Previsão Perfeita')
axes[1].set_title('Real vs Previsão', fontweight='bold')
axes[1].set_xlabel('Previsão (kW)')
axes[1].set_ylabel('Real (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos_modelos/arima_analise_erro.png', dpi=300, bbox_inches='tight')
plt.close()

print("Gráficos salvos:")
print("  - graficos_modelos/arima_validacao.png")
print("  - graficos_modelos/arima_teste.png")
print("  - graficos_modelos/arima_erro.png")
print("  - graficos_modelos/arima_analise_erro.png")

# [9] SALVAR RESULTADOS
print("\n[9] Salvando resultados...")

results_val = pd.DataFrame({
    'Date': val_series.index,
    'Real': val_series.values,
    'Previsao': val_predictions,
    'Erro': val_series.values - val_predictions
})
results_val.to_csv('arima_predictions_validation.csv', index=False)

results_test = pd.DataFrame({
    'Date': test_series.index,
    'Real': test_series.values,
    'Previsao': test_predictions,
    'Erro': errors
})
results_test.to_csv('arima_predictions_test.csv', index=False)

# Salvar modelo
import joblib
joblib.dump(final_result_test, 'arima_model.pkl')

print("Arquivos salvos:")
print("  - arima_predictions_validation.csv")
print("  - arima_predictions_test.csv")
print("  - arima_model.pkl")

print("\n" + "="*70)
print("MODELO ARIMA CONCLUÍDO")
print("="*70)
print(f"\n📊 Resumo final:")
print(f"   Parâmetros: {best_order} (sem sazonalidade)")
print(f"   Validação — MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
print(f"   Teste     — MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | MAPE: {test_mape:.2f}%")
print(f"\n✅ Próximo passo: Modelo LSTM")