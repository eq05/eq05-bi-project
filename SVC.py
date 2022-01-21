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

# --------------------------------
# ----- Variables del modelo -----
# --------------------------------
# ACCION = 'AAPL'
# FECHA_INICIO = '2019-01-01'
# FECHA_FINAL  = '2019-01-30'

TODAY = date.today().strftime("%Y-%m-%d")

def graficar_predicciones(dates, real, prediccion, label1, label2):
    fig, ax = plt.subplots()
    ax.plot(real[0:len(prediccion)],color='red', label=label1)
    ax.plot(prediccion, color='blue', label=label2)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    ax.legend()
    st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def load_data(accion, fecha_inicio, fecha_fin):
  dataframe = DataReader(
      accion, 
      data_source='yahoo', 
      start=fecha_inicio, 
      end=fecha_fin
  )
  return dataframe

def format_data(dataframe):
  dataframe["Date"] = dataframe.index
  df = pd.DataFrame({
      "Open": dataframe['Open'],
      "Volume": dataframe['Volume'],
      "High": dataframe['High'],
      "Low": dataframe['Low'],
      "Close": dataframe['Close'],
      "Date": dataframe['Date'].astype(str),
      "Adjclose": dataframe['Adj Close'],
  })

  # Changes The Date column as index columns
  df.index = pd.to_datetime(df['Date'])
  df
    
  # drop The original date column
  df = df.drop(['Date'], axis='columns')
  
  return df

def main():

  st.title('Proyecto Inteligencia de Negocios - SVC')
  stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
  selected_stock = st.selectbox('Selecciona la acción a evaluar', stocks)

  year_initial = st.slider('Año inicial:', 2010, 2020, 2020)
  date_initial = f"{year_initial}-01-01"

  data_load_state = st.text('Cargando datos...')
  dataframe = load_data(selected_stock, date_initial, TODAY)
  data_load_state.text('Cargando datos... ¡Listo!')

  df = format_data(dataframe)
  
  # Create predictor variables
  df['Open-Close'] = df.Open - df.Close
  df['High-Low'] = df.High - df.Low
    
  # Store all predictor variables in a variable X
  X = df[['Open-Close', 'High-Low']]

  # Target variables
  y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

  split_percentage = 0.8
  split = int(split_percentage*len(df))
    
  # Train data set
  X_train = X[:split]
  y_train = y[:split]
    
  # Test data set
  X_test = X[split:]
  y_test = y[split:]

  # Support vector classifier
  cls = SVC().fit(X_train, y_train)

  y_prediction = cls.predict(X_test)

  SCORE = precision_score(y_test, y_prediction, average='micro')
  conf_matrix = metrics.classification_report(y_test, y_prediction)
  st.markdown(f"""
  ## Matriz de Confusión:
```
${conf_matrix}
```
  ## Score:
  {SCORE*100}%
  """)


  df['Predicted_Signal'] = cls.predict(X)
  # Calculate daily returns
  df['Return'] = df.Close.pct_change()
  # Calculate strategy returns
  df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
  # Calculate Cumulutive returns
  df['Cum_Ret'] = df['Return'].cumsum()
  # Plot Strategy Cumulative returns 
  df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
  st.subheader("Columnas calculadas")
  st.write(df[['Open-Close', 'High-Low', "Predicted_Signal", "Return", "Strategy_Return", "Cum_Ret", "Cum_Strategy"]])


  st.subheader("Cum_Ret vs Cum_Strategy")
  graficar_predicciones([],df['Cum_Ret'], df['Cum_Strategy'], "Cum_Ret", "Cum_Strategy")

main()