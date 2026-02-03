import pandas as pd
import numpy as np

# Configurações de exibição
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Ler o arquivo CSV
# O separador é ponto e vírgula (;) e o decimal é ponto (.)
print("Lendo o arquivo CSV...")
df = pd.read_csv('household_power_consumption.txt', 
                 sep=';', 
                 low_memory=False,
                 na_values=['?', ''])

print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas")
print("\nPrimeiras linhas do dataset:")
print(df.head())

print("\nInformações sobre o dataset:")
print(df.info())

print("\nVerificando valores ausentes:")
print(df.isnull().sum())

# Salvar como Excel
print("\nConvertendo para Excel...")
output_file = 'household_power_consumption.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"\n✅ Arquivo Excel criado com sucesso: {output_file}")
print(f"📊 Total de registros: {len(df):,}")
print(f"📅 Período: {df['Date'].min()} até {df['Date'].max()}")

# Criar também uma versão com amostra (primeiras 10000 linhas) para facilitar testes
print("\nCriando também uma versão com amostra (10.000 linhas)...")
df_sample = df.head(10000)
df_sample.to_excel('household_power_consumption_sample.xlsx', index=False, engine='openpyxl')
print("✅ Arquivo de amostra criado: household_power_consumption_sample.xlsx")

print("\n" + "="*60)
print("CONVERSÃO CONCLUÍDA!")
print("="*60)