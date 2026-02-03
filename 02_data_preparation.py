import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PREPARAÇÃO DOS DADOS PARA MODELAGEM")
print("="*70)

# Carregar dados
print("\n[1] Carregando dados...")
df = pd.read_csv('energy_data_processed.csv', index_col=0, parse_dates=True)
print(f"Shape inicial: {df.shape}")

# MÉTODO 1: INTERPOLAÇÃO LINEAR - Tratamento de valores ausentes
print("\n[2] Aplicando Interpolação Linear para valores ausentes...")
missing_before = df['Global_active_power'].isnull().sum()
df['Global_active_power'] = df['Global_active_power'].interpolate(method='linear')
df['Global_active_power'].fillna(method='ffill', inplace=True)
df['Global_active_power'].fillna(method='bfill', inplace=True)
print(f"Valores ausentes tratados: {missing_before} → {df['Global_active_power'].isnull().sum()}")

# Tratar outras colunas numéricas
numerical_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
                  'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in numerical_cols:
    df[col] = df[col].interpolate(method='linear')
    df[col].fillna(method='ffill', inplace=True)
    df[col].fillna(method='bfill', inplace=True)

# MÉTODO 2: WINSORIZAÇÃO - Tratamento de outliers
print("\n[3] Aplicando Winsorização (percentis 1% e 99%)...")
p1 = df['Global_active_power'].quantile(0.01)
p99 = df['Global_active_power'].quantile(0.99)
outliers_before = ((df['Global_active_power'] < p1) | (df['Global_active_power'] > p99)).sum()
df['Global_active_power'] = df['Global_active_power'].clip(lower=p1, upper=p99)
print(f"Outliers limitados: {outliers_before} registros ajustados")
print(f"Limites: [{p1:.3f}, {p99:.3f}]")

# MÉTODO 3: AGREGAÇÃO HORÁRIA
print("\n[4] Agregando dados por hora...")
df_hourly = df.resample('H').agg({
    'Global_active_power': 'mean',
    'Global_reactive_power': 'mean',
    'Voltage': 'mean',
    'Global_intensity': 'mean',
    'Sub_metering_1': 'sum',
    'Sub_metering_2': 'sum',
    'Sub_metering_3': 'sum'
})
print(f"Dataset horário: {df_hourly.shape}")

# MÉTODO 4: FEATURES DE TEMPO + LAGS
print("\n[5] Criando features de tempo...")
df_hourly['hour'] = df_hourly.index.hour
df_hourly['day_of_week'] = df_hourly.index.dayofweek
df_hourly['month'] = df_hourly.index.month
df_hourly['day_of_year'] = df_hourly.index.dayofyear

print("\n[6] Criando features de lag...")
# Lags: 1 hora, 24 horas (1 dia), 168 horas (1 semana)
df_hourly['lag_1'] = df_hourly['Global_active_power'].shift(1)
df_hourly['lag_24'] = df_hourly['Global_active_power'].shift(24)
df_hourly['lag_168'] = df_hourly['Global_active_power'].shift(168)
print("Lags criados: 1h, 24h, 168h")

# Remover linhas com NaN dos lags
df_hourly_clean = df_hourly.dropna()
print(f"Dataset após remoção de NaN: {df_hourly_clean.shape}")

# MÉTODO 5: NORMALIZAÇÃO MINMAX (0-1)
print("\n[7] Aplicando MinMax Scaler (0-1)...")
scaler = MinMaxScaler()
df_hourly_clean['Global_active_power_scaled'] = scaler.fit_transform(
    df_hourly_clean[['Global_active_power']]
)
print("Normalização aplicada")

# MÉTODO 6: DIVISÃO 70/15/15
print("\n[8] Dividindo dados: 70% treino / 15% validação / 15% teste...")
train_size = int(len(df_hourly_clean) * 0.70)
val_size = int(len(df_hourly_clean) * 0.15)

train_data = df_hourly_clean[:train_size]
val_data = df_hourly_clean[train_size:train_size+val_size]
test_data = df_hourly_clean[train_size+val_size:]

print(f"Treino: {len(train_data)} ({len(train_data)/len(df_hourly_clean)*100:.1f}%)")
print(f"  Período: {train_data.index.min()} → {train_data.index.max()}")
print(f"Validação: {len(val_data)} ({len(val_data)/len(df_hourly_clean)*100:.1f}%)")
print(f"  Período: {val_data.index.min()} → {val_data.index.max()}")
print(f"Teste: {len(test_data)} ({len(test_data)/len(df_hourly_clean)*100:.1f}%)")
print(f"  Período: {test_data.index.min()} → {test_data.index.max()}")

# Visualização da divisão
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(train_data.index, train_data['Global_active_power'], label='Treino', alpha=0.7)
ax.plot(val_data.index, val_data['Global_active_power'], label='Validação', alpha=0.7)
ax.plot(test_data.index, test_data['Global_active_power'], label='Teste', alpha=0.7)
ax.set_title('Divisão dos Dados: Treino / Validação / Teste', fontweight='bold')
ax.set_xlabel('Data')
ax.set_ylabel('Potência Ativa (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_eda/divisao_dados.png', dpi=300, bbox_inches='tight')
plt.close()

# Salvar datasets
print("\n[9] Salvando datasets e scaler...")
df_hourly_clean.to_csv('data_hourly_prepared.csv')
train_data.to_csv('data_train.csv')
val_data.to_csv('data_validation.csv')
test_data.to_csv('data_test.csv')
joblib.dump(scaler, 'scaler_minmax.pkl')

print("\nArquivos salvos:")
print("  - data_hourly_prepared.csv")
print("  - data_train.csv")
print("  - data_validation.csv")
print("  - data_test.csv")
print("  - scaler_minmax.pkl")
print("  - graficos_eda/divisao_dados.png")

print("\n" + "="*70)
print("PREPARAÇÃO CONCLUÍDA")
print("="*70)