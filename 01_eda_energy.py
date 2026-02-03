import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*70)
print("ANÁLISE EXPLORATÓRIA DE DADOS - CONSUMO DE ENERGIA")
print("="*70)

# 1. CARREGAR OS DADOS
print("\n[1] CARREGANDO DADOS...")
df = pd.read_csv('household_power_consumption.txt', 
                 sep=';', 
                 low_memory=False,
                 na_values=['?', ''])

print(f"✅ Dataset carregado: {df.shape[0]:,} linhas e {df.shape[1]} colunas")

# 2. INFORMAÇÕES GERAIS
print("\n[2] INFORMAÇÕES GERAIS DO DATASET")
print("-" * 70)
print(df.info())

print("\n[3] PRIMEIRAS LINHAS:")
print("-" * 70)
print(df.head(10))

print("\n[4] ESTATÍSTICAS DESCRITIVAS:")
print("-" * 70)
print(df.describe())

# 3. VERIFICAR VALORES AUSENTES
print("\n[5] VALORES AUSENTES:")
print("-" * 70)
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Coluna': missing.index,
    'Valores Ausentes': missing.values,
    'Percentual (%)': missing_percent.values
})
print(missing_df)

# 4. CONVERTER DATA E HORA
print("\n[6] CONVERTENDO DATA E HORA...")
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                 format='%d/%m/%Y %H:%M:%S')
df.set_index('DateTime', inplace=True)

print(f"✅ Período dos dados: {df.index.min()} até {df.index.max()}")
print(f"✅ Duração total: {(df.index.max() - df.index.min()).days} dias")

# 5. CRIAR FEATURES TEMPORAIS
print("\n[7] CRIANDO FEATURES TEMPORAIS...")
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek  # 0=Segunda, 6=Domingo
df['DayName'] = df.index.day_name()
df['WeekOfYear'] = df.index.isocalendar().week

print("✅ Features criadas: Year, Month, Day, Hour, DayOfWeek, DayName, WeekOfYear")

# 6. VISUALIZAÇÕES
print("\n[8] GERANDO VISUALIZAÇÕES...")

# Criar pasta para salvar gráficos
import os
if not os.path.exists('graficos_eda'):
    os.makedirs('graficos_eda')

# 6.1 Consumo ao longo do tempo (amostra)
fig, ax = plt.subplots(figsize=(15, 6))
df['Global_active_power'].iloc[:10000].plot(ax=ax, linewidth=0.5)
ax.set_title('Consumo de Energia ao Longo do Tempo (Primeiras 10.000 observações)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Data/Hora')
ax.set_ylabel('Potência Ativa Global (kW)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos_eda/01_consumo_tempo.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 1 salvo: consumo_tempo.png")
plt.close()

# 6.2 Distribuição do consumo
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma
axes[0].hist(df['Global_active_power'].dropna(), bins=100, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribuição da Potência Ativa Global', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Potência Ativa (kW)')
axes[0].set_ylabel('Frequência')
axes[0].grid(True, alpha=0.3)

# Boxplot
axes[1].boxplot(df['Global_active_power'].dropna(), vert=True)
axes[1].set_title('Boxplot da Potência Ativa Global', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Potência Ativa (kW)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos_eda/02_distribuicao_consumo.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 2 salvo: distribuicao_consumo.png")
plt.close()

# 6.3 Consumo médio por hora do dia
hourly_avg = df.groupby('Hour')['Global_active_power'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
hourly_avg.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Consumo Médio de Energia por Hora do Dia', fontsize=14, fontweight='bold')
ax.set_xlabel('Hora do Dia')
ax.set_ylabel('Potência Ativa Média (kW)')
ax.set_xticklabels(range(24), rotation=0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graficos_eda/03_consumo_por_hora.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 3 salvo: consumo_por_hora.png")
plt.close()

# 6.4 Consumo médio por dia da semana
dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
weekly_avg = df.groupby('DayOfWeek')['Global_active_power'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
weekly_avg.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_title('Consumo Médio de Energia por Dia da Semana', fontsize=14, fontweight='bold')
ax.set_xlabel('Dia da Semana')
ax.set_ylabel('Potência Ativa Média (kW)')
ax.set_xticklabels(dias_semana, rotation=45)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graficos_eda/04_consumo_por_dia_semana.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 4 salvo: consumo_por_dia_semana.png")
plt.close()

# 6.5 Consumo médio por mês
monthly_avg = df.groupby('Month')['Global_active_power'].mean()
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
         'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

fig, ax = plt.subplots(figsize=(12, 6))
monthly_avg.plot(kind='bar', ax=ax, color='green', edgecolor='black')
ax.set_title('Consumo Médio de Energia por Mês', fontsize=14, fontweight='bold')
ax.set_xlabel('Mês')
ax.set_ylabel('Potência Ativa Média (kW)')
ax.set_xticklabels(meses, rotation=45)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graficos_eda/05_consumo_por_mes.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 5 salvo: consumo_por_mes.png")
plt.close()

# 6.6 Heatmap de consumo por hora e dia da semana
heatmap_data = df.pivot_table(values='Global_active_power', 
                               index='Hour', 
                               columns='DayOfWeek', 
                               aggfunc='mean')

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
            ax=ax, cbar_kws={'label': 'Potência Ativa (kW)'})
ax.set_title('Mapa de Calor: Consumo por Hora e Dia da Semana', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Dia da Semana')
ax.set_ylabel('Hora do Dia')
ax.set_xticklabels(dias_semana)
plt.tight_layout()
plt.savefig('graficos_eda/06_heatmap_hora_dia.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 6 salvo: heatmap_hora_dia.png")
plt.close()

# 6.7 Matriz de correlação
numerical_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

correlation_matrix = df[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlação'})
ax.set_title('Matriz de Correlação entre Variáveis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos_eda/07_matriz_correlacao.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico 7 salvo: matriz_correlacao.png")
plt.close()

# 7. ESTATÍSTICAS IMPORTANTES
print("\n[9] ESTATÍSTICAS IMPORTANTES:")
print("-" * 70)
print(f"Consumo médio diário: {df['Global_active_power'].mean():.3f} kW")
print(f"Consumo máximo: {df['Global_active_power'].max():.3f} kW")
print(f"Consumo mínimo: {df['Global_active_power'].min():.3f} kW")
print(f"Desvio padrão: {df['Global_active_power'].std():.3f} kW")

print("\n[10] INSIGHTS PRELIMINARES:")
print("-" * 70)
print(f"📊 Hora de maior consumo médio: {hourly_avg.idxmax()}h ({hourly_avg.max():.3f} kW)")
print(f"📊 Hora de menor consumo médio: {hourly_avg.idxmin()}h ({hourly_avg.min():.3f} kW)")
print(f"📊 Dia da semana com maior consumo: {dias_semana[weekly_avg.idxmax()]} ({weekly_avg.max():.3f} kW)")
print(f"📊 Mês com maior consumo: {meses[monthly_avg.idxmax()-1]} ({monthly_avg.max():.3f} kW)")

# 8. SALVAR DATASET PROCESSADO
print("\n[11] SALVANDO DATASET PROCESSADO...")
df.to_csv('energy_data_processed.csv')
print("✅ Dataset processado salvo: energy_data_processed.csv")

print("\n" + "="*70)
print("ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
print("="*70)
print(f"\n📁 {7} gráficos salvos na pasta 'graficos_eda/'")
print("📁 Dataset processado salvo: 'energy_data_processed.csv'")
print("\n✅ Próximo passo: Preparação dos dados para modelagem")