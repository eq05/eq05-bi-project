# Importaciones generales
import streamlit as st
import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

# Para cargar los datos de Yahoo Finance 
from pandas_datareader.data import DataReader

# Para los modelos
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from plotly import graph_objs as go

import warnings
warnings.filterwarnings("ignore")

TODAY = date.today().strftime("%Y-%m-%d")

def graficar_predicciones(dates, real, prediccion):
    fig, ax = plt.subplots()
    ax.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
    ax.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    ax.legend()
    st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def load_data(ACCION, FECHA_INICIO, FECHA_FINAL):
  dataset = DataReader(
    ACCION, 
    data_source='yahoo', 
    start=FECHA_INICIO, 
    end=FECHA_FINAL
  )
  return dataset


def main():
  st.title('Proyecto Inteligencia de Negocios - LSTM')
  st.markdown("""
  ## Instrucciones:
  1. Seleccione la acción que desea evaluar en el modelo, los datos se cargarán de Yahoo Finance
  2. Seleccione el año inicial, la data se cargara desde 1 de enero del año seleccionado hasta la fecha actual
  3. Configure las unidades y epocas del modelo LSTM, luego espere y vea los resultados.

  **Nota:**   
  Los datos se cargarán desde el año inicial que escoja hasta la fecha actual, y el 50% se usará para entrenamiento y el otro 50% para la validación
  """)

  st.header('Paso 1 - Seleccione la acción')
  stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'BVN')
  selected_stock = st.selectbox('Selecciona la acción a evaluar', stocks)

  st.header('Paso 2 - Seleccione el año inicial')
  year_initial = st.slider('Año inicial:', 2014, 2020, 2020)
  date_initial = f"{year_initial}-01-01"

  year_middle = year_initial + int((2021 - year_initial)/2)

  data_load_state = st.text('Cargando datos...')
  dataframe = load_data(selected_stock, date_initial, TODAY)
  # dataframe = load_data(selected_stock, '2016-01-01', '2017-12-01')
  data_load_state.text('Cargando datos... ¡Listo!')

  st.subheader('Datos cargados:')
  st.write(dataframe)

  set_entrenamiento = dataframe[:f'{year_middle}'].iloc[:,0:1]
  set_validacion = dataframe[f'{year_middle+1}':].iloc[:,0:1]

  sc = MinMaxScaler(feature_range=(0,1))
  set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

  st.header('Paso 3 - Configurar parámetros')

  # La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
  # partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
  # time_step = 60
  st.subheader('Parametros de entrenamiento')
  time_step = st.number_input('Tamaño de los grupos de entrenamiento (Recomendado: 60)', 30, 100, 60, 10)

  X_train = []
  Y_train = []
  m = len(set_entrenamiento_escalado)

  for i in range(time_step,m):
      # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
      X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

      # Y: el siguiente dato
      Y_train.append(set_entrenamiento_escalado[i,0])
  X_train, Y_train = np.array(X_train), np.array(Y_train)

  # Reshape X_train para que se ajuste al modelo en Keras
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  #
  # Red LSTM
  #
  dim_entrada = (X_train.shape[1],1)
  dim_salida = 1
  st.subheader('Parametros para Keras')
  na = st.number_input('Neuronas (Recomendado: 50) ', 1, 50, 4, 1)
  epocas = st.number_input('Epocas de entrenamiento (Recomendado: 20) ', 1, 20, 4, 1)

  if st.button('Ejecutar modelo'):
    data_train_state = st.text('Entrenando modelo...')
    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    modelo.add(Dense(units=dim_salida))
    modelo.compile(optimizer='rmsprop', loss='mse')
    modelo.fit(X_train,Y_train,epochs=epocas,batch_size=32)
    data_train_state.text('Entrenando modelo... ¡Listo!')

    #
    # Validación (predicción del valor de las acciones)
    #
    x_test = set_validacion.values
    x_test = sc.transform(x_test)

    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)

    st.header('Resultados')
    # Graficar resultados
    graficar_predicciones(set_validacion.index,set_validacion.values,prediccion)
  
  else:
    st.markdown("__Click en ejecutar para correr el modelo__")


main()