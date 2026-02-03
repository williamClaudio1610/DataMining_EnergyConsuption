import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import joblib
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Previsão de Demanda de Energia",
    page_icon="⚡",
    layout="wide"
)

# Título principal
st.title("Dashboard - Previsão de Demanda de Energia")
st.markdown("---")

# Carregar dados e modelos
@st.cache_data
def load_data():
    train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)
    val = pd.read_csv('data_validation.csv', index_col=0, parse_dates=True)
    test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)
    arima_pred = pd.read_csv('arima_predictions_test.csv', parse_dates=['Date'])
    rnn_pred = pd.read_csv('rnn_predictions_test.csv', parse_dates=['Date'])
    return train, val, test, arima_pred, rnn_pred

@st.cache_resource
def load_models():
    arima_model = joblib.load('arima_model.pkl')
    rnn_model = keras.models.load_model('rnn_model.h5', compile=False)
    scaler_X = joblib.load('scaler_X_rnn.pkl')
    scaler_y = joblib.load('scaler_y_rnn.pkl')
    scaler_minmax = joblib.load('scaler_minmax.pkl')
    return arima_model, rnn_model, scaler_X, scaler_y, scaler_minmax

# Carregar dados
with st.spinner('Carregando dados e modelos...'):
    train, val, test, arima_pred, rnn_pred = load_data()
    arima_model, rnn_model, scaler_X, scaler_y, scaler_minmax = load_models()

st.success('Dados e modelos carregados!')

# Sidebar - Menu de navegação
st.sidebar.title("Menu")
opcao = st.sidebar.radio(
    "Selecione uma opção:",
    ["📈 Visão Geral", "🔮 Fazer Previsão", "📊 Comparação de Modelos", "📉 Análise de Resultados"]
)

# ============================================================
# OPÇÃO 1: VISÃO GERAL
# ============================================================
if opcao == "📈 Visão Geral":
    st.header("📈 Visão Geral dos Dados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", f"{len(train) + len(val) + len(test):,}")
    with col2:
        st.metric("Período", f"{train.index.min().date()} - {test.index.max().date()}")
    with col3:
        st.metric("Consumo Médio", f"{test['Global_active_power'].mean():.2f} kW")
    
    st.markdown("---")
    
    # Gráfico: Dados ao longo do tempo
    st.subheader("Consumo de Energia ao Longo do Tempo")
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(train.index, train['Global_active_power'], label='Treino', alpha=0.7, linewidth=0.5)
    ax.plot(val.index, val['Global_active_power'], label='Validação', alpha=0.7, linewidth=0.5)
    ax.plot(test.index, test['Global_active_power'], label='Teste', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Data')
    ax.set_ylabel('Potência Ativa (kW)')
    ax.set_title('Divisão dos Dados: Treino / Validação / Teste')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Estatísticas descritivas
    st.subheader("📊 Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Conjunto de Teste:**")
        st.dataframe(test['Global_active_power'].describe())
    
    with col2:
        # Distribuição
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(test['Global_active_power'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Potência Ativa (kW)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do Consumo (Teste)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ============================================================
# OPÇÃO 2: FAZER PREVISÃO
# ============================================================
elif opcao == "🔮 Fazer Previsão":
    st.header("🔮 Fazer Nova Previsão")
    
    st.write("Insira os valores para fazer uma previsão de consumo de energia:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consumo_atual = st.number_input("Consumo Atual (kW)", 
                                         min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        lag_1 = st.number_input("Consumo 1h atrás (kW)", 
                                min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    with col2:
        lag_24 = st.number_input("Consumo 24h atrás (kW)", 
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        lag_168 = st.number_input("Consumo 168h atrás (kW)", 
                                   min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    with col3:
        hora = st.slider("Hora do Dia", 0, 23, 12)
        dia_semana = st.selectbox("Dia da Semana", 
                                   ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'])
        mes = st.slider("Mês", 1, 12, 6)
    
    dia_semana_map = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 
                      'Sexta': 4, 'Sábado': 5, 'Domingo': 6}
    
    modelo_escolhido = st.radio("Escolha o modelo:", ["ARIMA", "RNN"])
    
    if st.button("🔮 Fazer Previsão", type="primary"):
        
        if modelo_escolhido == "RNN":
            # Preparar input para RNN
            input_data = np.array([[consumo_atual, lag_1, lag_24, lag_168, 
                                    hora, dia_semana_map[dia_semana], mes]])
            
            # Normalizar
            input_scaled = scaler_X.transform(input_data)
            input_rnn = input_scaled.reshape((1, 1, input_scaled.shape[1]))
            
            # Prever
            pred_scaled = rnn_model.predict(input_rnn, verbose=0)
            previsao = scaler_y.inverse_transform(pred_scaled)[0][0]
            
            st.success(f"✅ Previsão RNN: **{previsao:.3f} kW**")
            
        else:  # ARIMA
            # ARIMA usa apenas a série temporal
            st.info("ℹ️ ARIMA usa apenas valores históricos da série temporal")
            st.warning("Para ARIMA, use a opção 'Análise de Resultados' para ver previsões")
        
        # Visualização
        st.markdown("---")
        st.subheader("📊 Contexto da Previsão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras dos inputs
            fig, ax = plt.subplots(figsize=(8, 5))
            valores = [consumo_atual, lag_1, lag_24, lag_168]
            labels = ['Atual', '1h atrás', '24h atrás', '168h atrás']
            ax.bar(labels, valores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel('Potência (kW)')
            ax.set_title('Histórico de Consumo')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            # Padrão médio por hora
            fig, ax = plt.subplots(figsize=(8, 5))
            hourly_avg = test.groupby('hour')['Global_active_power'].mean()
            ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
            ax.axvline(x=hora, color='red', linestyle='--', linewidth=2, label=f'Hora atual: {hora}h')
            ax.set_xlabel('Hora do Dia')
            ax.set_ylabel('Consumo Médio (kW)')
            ax.set_title('Padrão Médio de Consumo por Hora')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============================================================
# OPÇÃO 3: COMPARAÇÃO DE MODELOS
# ============================================================
elif opcao == "📊 Comparação de Modelos":
    st.header("📊 Comparação de Modelos: ARIMA vs RNN")
    
    # Métricas
    arima_mae = mean_absolute_error(arima_pred['Real'], arima_pred['Previsao'])
    arima_rmse = np.sqrt(np.mean((arima_pred['Real'] - arima_pred['Previsao'])**2))
    
    rnn_mae = mean_absolute_error(rnn_pred['Real'], rnn_pred['Previsao'])
    rnn_rmse = np.sqrt(np.mean((rnn_pred['Real'] - rnn_pred['Previsao'])**2))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 ARIMA")
        st.metric("MAE", f"{arima_mae:.4f} kW")
        st.metric("RMSE", f"{arima_rmse:.4f} kW")
    
    with col2:
        st.subheader("🤖 RNN")
        st.metric("MAE", f"{rnn_mae:.4f} kW", delta=f"{rnn_mae - arima_mae:.4f} kW")
        st.metric("RMSE", f"{rnn_rmse:.4f} kW", delta=f"{rnn_rmse - arima_rmse:.4f} kW")
    
    st.markdown("---")
    
    # Gráfico comparativo
    st.subheader("📈 Comparação Visual (primeiras 500 horas)")
    
    plot_range = min(500, len(arima_pred))
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # ARIMA
    axes[0].plot(arima_pred['Date'][:plot_range], arima_pred['Real'][:plot_range], 
                 label='Real', linewidth=2)
    axes[0].plot(arima_pred['Date'][:plot_range], arima_pred['Previsao'][:plot_range], 
                 label='ARIMA', linewidth=2, alpha=0.7)
    axes[0].set_ylabel('Potência (kW)')
    axes[0].set_title('ARIMA - Real vs Previsão')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RNN
    axes[1].plot(rnn_pred['Date'][:plot_range], rnn_pred['Real'][:plot_range], 
                 label='Real', linewidth=2)
    axes[1].plot(rnn_pred['Date'][:plot_range], rnn_pred['Previsao'][:plot_range], 
                 label='RNN', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Data')
    axes[1].set_ylabel('Potência (kW)')
    axes[1].set_title('RNN - Real vs Previsão')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Tabela comparativa
    st.markdown("---")
    st.subheader("📋 Tabela Comparativa de Métricas")
    
    comparacao = pd.DataFrame({
        'Modelo': ['ARIMA', 'RNN'],
        'MAE (kW)': [arima_mae, rnn_mae],
        'RMSE (kW)': [arima_rmse, rnn_rmse],
        'Melhoria vs ARIMA (%)': [0, ((arima_mae - rnn_mae) / arima_mae) * 100]
    })
    st.dataframe(comparacao, hide_index=True)

# ============================================================
# OPÇÃO 4: ANÁLISE DE RESULTADOS
# ============================================================
elif opcao == "📉 Análise de Resultados":
    st.header("📉 Análise Detalhada de Resultados")
    
    modelo_analise = st.selectbox("Selecione o modelo para análise:", ["ARIMA", "RNN"])
    
    if modelo_analise == "ARIMA":
        df_pred = arima_pred.copy()
    else:
        df_pred = rnn_pred.copy()
    
    # Estatísticas dos erros
    st.subheader("📊 Estatísticas dos Erros")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Erro Médio", f"{df_pred['Erro'].mean():.4f} kW")
    with col2:
        st.metric("Desvio Padrão", f"{df_pred['Erro'].std():.4f} kW")
    with col3:
        st.metric("Erro Mínimo", f"{df_pred['Erro'].min():.4f} kW")
    with col4:
        st.metric("Erro Máximo", f"{df_pred['Erro'].max():.4f} kW")
    
    st.markdown("---")
    
    # Gráficos de análise
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição dos erros
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_pred['Erro'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Erro (kW)')
        ax.set_ylabel('Frequência')
        ax.set_title(f'Distribuição dos Erros - {modelo_analise}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Real vs Previsão (scatter)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df_pred['Previsao'], df_pred['Real'], alpha=0.5, s=10)
        ax.plot([df_pred['Real'].min(), df_pred['Real'].max()], 
                [df_pred['Real'].min(), df_pred['Real'].max()], 
                'r--', linewidth=2, label='Previsão Perfeita')
        ax.set_xlabel('Previsão (kW)')
        ax.set_ylabel('Real (kW)')
        ax.set_title(f'Real vs Previsão - {modelo_analise}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Tabela de dados
    st.markdown("---")
    st.subheader("📋 Tabela de Resultados (primeiras 100 linhas)")
    st.dataframe(df_pred.head(100), hide_index=True)
    
    # Download dos resultados
    csv = df_pred.to_csv(index=False)
    st.download_button(
        label="📥 Download Resultados CSV",
        data=csv,
        file_name=f'{modelo_analise.lower()}_resultados.csv',
        mime='text/csv',
    )

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dashboard de Previsão de Demanda de Energia | ISPTEC 2025/2026</p>
        <p>Mineração de Dados - Projeto Final</p>
    </div>
    """,
    unsafe_allow_html=True
)