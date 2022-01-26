# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

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

def plot_heatmap(y_test, prediccion):
  # Heatmap
  cm = confusion_matrix(y_test, prediccion)
  fig, ax = plt.subplots()
  ax.imshow(cm)
  ax.grid(False)
  ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
  ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
  ax.set_ylim(1.5, -0.5)
  for i in range(2):
      for j in range(2):
          ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
  st.pyplot(fig)


def preprocesar(dataframe):
  df = dataframe.copy()
  size = len(df)

  df = df.reset_index()
  for index,row in df.iterrows():
    if index < (size - 1):
      df.loc[index,'High Dia Sgte'] = df.loc[index+1,'High']
      df.loc[index,'Low Dia Sgte'] = df.loc[index+1,'Low']

  df.loc[size - 1,'High Dia Sgte'] = df.loc[size - 1,'High']
  df.loc[size - 1,'Low Dia Sgte'] = df.loc[size - 1,'Low']

  window = 5
  df['HH'] = df['High Dia Sgte'].rolling(window = window).max()
  df['LL'] = df['Low Dia Sgte'].rolling(window = window).min()
  df['T'] = np.where(df['High Dia Sgte'] == df['HH'],1,np.nan)
  df['T'] = np.where(df['Low Dia Sgte'] == df['LL'],0,df['T'])
  df['CompraVenta'] = df['T'].ffill().fillna(1.0)

  return df[list(dataframe.columns) + ['CompraVenta']]

def main():

  st.subheader('Selecciona el dataset')
  df, success = select_dataset()
  
  if not success: return

  st.subheader('Datos cargados:')
  st.write(df)
  st.info(f"Tamaño del dataset: {len(df)}")

  st.subheader('Compra/Venta')
  df = preprocesar(df)
  st.write(df)

  # Variables de predicción y a predecir
  X = df[["Open", "High", "Low", "Close"]]
  y = df["CompraVenta"]

  test_size = st.number_input("Porcentaje de datos para test", 0.05, 0.99, 0.30)
  st.markdown(f"""
    * **Tamaño test = {int(len(df) * test_size)}**
    * **Tamaño train = {int(len(df) * (1- test_size))}**
  """)
  random_state = st.number_input("Random State", 10, 200, 101)
  X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = test_size, random_state = random_state)

  # Predicción
  prediction_loading = st.text('Cargando predicción...')
  modelo = LogisticRegression() # Modelo de predicción
  modelo.fit(X_train, y_train) # Entrenamiento
  prediccion = modelo.predict(X_test) # Predicción
  prediction_loading.text('Cargando predicción... ¡Listo!')

  st.write(pd.DataFrame({ 'Real': y_test, 'Prediccion': prediccion }))

  st.header("Validación")
  st.subheader("Matriz de confución")
  plot_heatmap(y_test, prediccion)
  st.subheader("Classification Report")
  st.code(classification_report(y_test, prediccion))

main()