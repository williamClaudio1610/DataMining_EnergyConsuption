import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODELO RNN - PREVISÃO DE DEMANDA DE ENERGIA")
print("="*70)

# [1] CARREGAR DADOS
print("\n[1] Carregando dados...")
train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)
val = pd.read_csv('data_validation.csv', index_col=0, parse_dates=True)
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)

# ============================================================
# FEATURES: Global_active_power + Lags (1h, 24h, 168h)
# ============================================================
features = ['Global_active_power', 'lag_1', 'lag_24', 'lag_168', 
            'hour', 'day_of_week', 'month']
target = 'Global_active_power'

X_train = train[features].values
y_train = train[target].values
X_val = val[features].values
y_val = val[target].values
X_test = test[features].values
y_test = test[target].values

print(f"Treino: {X_train.shape}")
print(f"Validação: {X_val.shape}")
print(f"Teste: {X_test.shape}")

# ============================================================
# NORMALIZAÇÃO: MinMax Scaler (0-1)
# ============================================================
print("\n[2] Normalizando dados...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# ============================================================
# RESHAPE PARA RNN: (samples, timesteps, features)
# timesteps = 1 (cada registro é independente)
# ============================================================
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_rnn = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ============================================================
# ARQUITETURA RNN: 2 camadas SimpleRNN + Dropout + Dense
# ============================================================
print("\n[3] Construindo modelo RNN...")
model = Sequential([
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.2),
    SimpleRNN(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# ============================================================
# TREINAMENTO: EarlyStopping para evitar overfitting
# ============================================================
print("\n[4] Treinando modelo RNN...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_rnn, y_train_scaled,
    validation_data=(X_val_rnn, y_val_scaled),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# [5] PREVISÕES
print("\n[5] Fazendo previsões...")
val_pred_scaled = model.predict(X_val_rnn, verbose=0)
test_pred_scaled = model.predict(X_test_rnn, verbose=0)

# Desnormalizar
val_pred = scaler_y.inverse_transform(val_pred_scaled).flatten()
test_pred = scaler_y.inverse_transform(test_pred_scaled).flatten()

# ============================================================
# MÉTRICAS: MAE + RMSE
# ============================================================
val_mae = mean_absolute_error(y_val, val_pred)
val_rmse = np.sqrt(np.mean((y_val - val_pred)**2))

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(np.mean((y_test - test_pred)**2))

print(f"\n--- Resultados Validação ---")
print(f"MAE:  {val_mae:.4f} kW")
print(f"RMSE: {val_rmse:.4f} kW")

print(f"\n--- Resultados Teste ---")
print(f"MAE:  {test_mae:.4f} kW")
print(f"RMSE: {test_rmse:.4f} kW")

# [6] GRÁFICOS
print("\n[6] Gerando gráficos...")

# Loss durante treinamento
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history['loss'], label='Treino')
ax.plot(history.history['val_loss'], label='Validação')
ax.set_title('RNN - Loss durante Treinamento', fontweight='bold')
ax.set_xlabel('Época')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/rnn_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# Previsões validação (primeiras 500h)
fig, ax = plt.subplots(figsize=(15, 6))
plot_range = min(500, len(y_val))
ax.plot(val.index[:plot_range], y_val[:plot_range], label='Real', linewidth=2)
ax.plot(val.index[:plot_range], val_pred[:plot_range], label='Previsão RNN', linewidth=2, alpha=0.7)
ax.set_title('RNN - Validação: Real vs Previsão (primeiras 500h)', fontweight='bold')
ax.set_xlabel('Data')
ax.set_ylabel('Potência Ativa (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/rnn_validacao.png', dpi=300, bbox_inches='tight')
plt.close()

# Previsões teste (primeiras 500h)
fig, ax = plt.subplots(figsize=(15, 6))
plot_range = min(500, len(y_test))
ax.plot(test.index[:plot_range], y_test[:plot_range], label='Real', linewidth=2)
ax.plot(test.index[:plot_range], test_pred[:plot_range], label='Previsão RNN', linewidth=2, alpha=0.7)
ax.set_title('RNN - Teste: Real vs Previsão (primeiras 500h)', fontweight='bold')
ax.set_xlabel('Data')
ax.set_ylabel('Potência Ativa (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_modelos/rnn_teste.png', dpi=300, bbox_inches='tight')
plt.close()

print("Gráficos salvos em graficos_modelos/")

# [7] SALVAR RESULTADOS
print("\n[7] Salvando resultados...")
results_test = pd.DataFrame({
    'Date': test.index,
    'Real': y_test,
    'Previsao': test_pred,
    'Erro': y_test - test_pred
})
results_test.to_csv('rnn_predictions_test.csv', index=False)

model.save('rnn_model.h5')

import joblib
joblib.dump(scaler_X, 'scaler_X_rnn.pkl')
joblib.dump(scaler_y, 'scaler_y_rnn.pkl')

print("Arquivos salvos:")
print("  - rnn_predictions_test.csv")
print("  - rnn_model.h5")
print("  - scaler_X_rnn.pkl")
print("  - scaler_y_rnn.pkl")

print("\n" + "="*70)
print("MODELO RNN CONCLUÍDO")
print("="*70)
print(f"\n📊 Resumo final:")
print(f"   Arquitetura: 2 camadas SimpleRNN (50 unidades cada)")
print(f"   Validação — MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")
print(f"   Teste     — MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")