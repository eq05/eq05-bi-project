# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date, datetime

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
import pandas as pd 
import numpy as np
from pandas_datareader.data import DataReader
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# Modelo
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from interactions import select_dataset 

import calendar

def main():
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.header("SVR")

  st.subheader('Selecciona el dataset')
  selected_stock = st.text_input('Ingresa la acción', "AAPL")
  agno = st.slider("Año", 2010, datetime.now().year, 2020)
  mes = st.slider("Mes", 1, 12, 2)


  # start = datetime(agno, mes, 1).date
  # end = datetime(agno, mes, 31).date

  last_day = calendar.monthrange(agno, mes)[1]
  try:
    df = yf.download(selected_stock, f'{agno}-{mes}-1', f'{agno}-{mes}-{last_day}')
  except:
    st.warning("Ingrese una acción o fechas valida")
    return


  if len(df)<10:  
    st.warning("Ingrese una acción o fechas valida")
    return

  date_str = f"Mes {mes} del año {agno}"

  st.subheader('Datos cargados:')
  st.markdown(f">Fecha: **{date_str}**")
  st.write(df)
  st.info(f"Tamaño del dataset: {len(df)}")

  df["Date"] = df.index
  df['Date'] = df['Date'].apply(lambda d: d.strftime("%Y-%m-%d"))

  #Mostrar la última fila de la data
  actual_price = df.tail(1)

  st.subheader('Precio Actual')
  st.write(actual_price)

  df = df.head(len(df)-1)

  #Crear listas vacías para almacenar la data independiente y dependiente
  days = list()
  adj_close_prices = list()

  #Obtener la fecha y el precio de cierre ajustado
  df_days = df.loc[:, 'Date']
  df_adj_close = df.loc[:, 'Adj Close']

  #Crear el data set independiente
  for day in df_days:
      days.append([int(day.split('-')[2])])
      
  #Crear el data set dependiente
  for adj_close_price in df_adj_close:
      adj_close_prices.append( float(adj_close_price) )

  st.subheader("Creación de modelos Support Vector Regression")
  st.code("""
#Crear y entrenar un modelo SVR usando linear kernel
lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, adj_close_prices)

#Crear y entrenar un modelo SVR usando polynomial kernel
poly_svr = SVR(kernel='poly', degree=2)
poly_svr.fit(days, adj_close_prices)

#Crear y entrenar un modelo SVR usando rbf kernel
rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
rbf_svr.fit(days, adj_close_prices)
  """)
  #Crear los modelos de Support Vector Regression

  #Crear y entrenar un modelo SVR usando linear kernel
  lin_svr = SVR(kernel='linear', C=1000.0)
  lin_svr.fit(days, adj_close_prices)

  #Crear y entrenar un modelo SVR usando polynomial kernel
  poly_svr = SVR(kernel='poly', degree=2)
  poly_svr.fit(days, adj_close_prices)

  #Crear y entrenar un modelo SVR usando rbf kernel
  rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
  rbf_svr.fit(days, adj_close_prices)

  #Mostrar los modelos en un gráfico para ver cuál tiene el mejor ajuste a la data original
  plt.figure(figsize=(16, 8))
  plt.scatter(days, adj_close_prices, color='red', label='Data')
  plt.plot(days, rbf_svr.predict(days), color='green', label='RBF model')
  plt.plot(days, poly_svr.predict(days), color='orange', label='Polynomial model')
  plt.plot(days, lin_svr.predict(days), color='blue', label='Linear model')
  plt.legend()
  st.pyplot()

  st.subheader('Predicción del precio para un día dado')
  st.markdown(f"## `{selected_stock}`")
  st.markdown(f">Fecha: **{date_str}**")

  in_day = st.number_input(f"Día (Rango: [1-{last_day}])", 2, last_day, last_day, key='svreq5uniq')
  day = [[int(in_day)]]
  day_ayer = [[int(in_day-1)]]

  col1, col2, col3 = st.columns(3)
  col1.metric(
    'RBF SVR', 
    round(rbf_svr.predict(day)[0], 3), 
    round(rbf_svr.predict(day)[0] - rbf_svr.predict(day_ayer)[0], 3)
    )
  col2.metric(
    "Linear SVR", 
    round(lin_svr.predict(day)[0], 3), 
    round(lin_svr.predict(day)[0] - lin_svr.predict(day_ayer)[0], 3)
    )
  col3.metric(
    "Polynomial SVR", 
    round(poly_svr.predict(day)[0], 3), 
    round(poly_svr.predict(day)[0] - poly_svr.predict(day_ayer)[0], 3)
    )

  st.subheader('Precios')
  st.write(df['Adj Close'])

  st.subheader('Precio Actual')
  st.write(actual_price['Adj Close'])

# main()