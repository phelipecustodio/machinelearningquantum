# Importe as bibliotecas necessárias.
import numpy as np
import pandas as pd
import time
import requests
import streamlit as st
import streamlit.components.v1 as components
from xgboost import XGBClassifier
import pickle

# Função para carregar o modelo e os codificadores a partir do GitHub
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        st.error(f"Erro ao carregar dados do GitHub. Status code: {response.status_code}")
        st.stop()

# Carregue o modelo e os codificadores
loaded_model = load_data_from_github("https://github.com/phelipecustodio/machinelearningquantum/raw/main/credit_score_multi_class_xgboost_model.json")
loaded_le = load_data_from_github("https://github.com/phelipecustodio/machinelearningquantum/raw/main/credit_score_multi_class_le.pkl")
loaded_enc = load_data_from_github("https://github.com/phelipecustodio/machinelearningquantum/raw/main/credit_score_multi_class_ord_encoder.pkl")

# Defina a função para obter os dados de entrada do usuário
def user_input_data():
    Total_EMI_per_month    = st.slider('Total EMI per month (EMI total por mês)', 0.0, 1780.0, 107.0, 0.1)
    Num_Bank_Accounts      = st.slider('Número de contas bancárias', 0, 11, 5, 1)
    Num_of_Delayed_Payment = st.slider('Número de pagamentos atrasados', 0, 25, 14, 1) 
    Delay_from_due_date    = st.slider('Atraso a partir da data de vencimento', 0, 62, 21, 1)
    Changed_Credit_Limit   = st.slider('Limite de crédito alterado', 0.5, 30.0, 9.40, 0.1)
    Num_Credit_Card        = st.slider('Número de cartões de crédito', 0, 11, 5, 1)
    Outstanding_Debt       = st.slider('Dívida pendente', 0.0, 5000.0, 1426.0, 0.1)
    Interest_Rate          = st.slider('Taxa de juros', 1, 34, 14, 1)   
    Credit_Mix             = st.selectbox('Mix de crédito:', ['Padrão', 'Ruim', 'Bom'])
    
    data = { 
        'Total_EMI_per_month'   : Total_EMI_per_month,
        'Num_Bank_Accounts'     : Num_Bank_Accounts,
        'Num_of_Delayed_Payment': Num_of_Delayed_Payment, 
        'Delay_from_due_date'   : Delay_from_due_date,
        'Changed_Credit_Limit'  : Changed_Credit_Limit,
        'Num_Credit_Card'       : Num_Credit_Card,        
        'Outstanding_Debt'      : Outstanding_Debt,
        'Interest_Rate'         : Interest_Rate,       
        'Credit_Mix'            : Credit_Mix,
    }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data

# Defina o layout
st.set_page_config(layout="wide")

# Cabeçalho
st.markdown("<h2 style='text-align:center; color:#FF69B4;'>Classificação de Pontuação de Crédito</h2>", unsafe_allow_html=True)

# Barra lateral
st.sidebar.header("Parâmetros de entrada do usuário")
df = user_input_data() 

# Mostre as entradas do usuário
if st.sidebar.checkbox('Mostrar entradas do usuário:', value=True):
    st.sidebar.dataframe(df.astype(str).T.rename(columns={0:'input_data'}).style.highlight_max(axis=0))

# Botão de Previsão
if st.sidebar.button('Fazer Previsão'):   
    # Use o modelo carregado para fazer previsões
    df[cat]    = loaded_enc.transform(df[cat]) 
    prediction = loaded_model.predict(df)
    prediction = loaded_le.inverse_transform(prediction)[0]

    st.success(f'A probabilidade de pontuação de crédito é:&emsp;{prediction}')
