# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data(data):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Datos de series de tiempo con RangeSlider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def main():
  st.title('Proyecto Inteligencia de Negocios - Facebook Prophet')

  # stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'BVN')
  # selected_stock = st.selectbox('Selecciona la acción a evaluar', stocks)
  selected_stock = st.text_input('Ingresa la acción', "AAPL")

  n_years = st.slider('Años de predicción:', 1, 4)
  period = n_years * 365
    
  data_load_state = st.text('Cargando datos...')
  data = load_data(selected_stock)
  data_load_state.text('Cargando datos... ¡Listo!')

  st.subheader('Datos cargados:')
  st.write(data.tail())
    
  plot_raw_data(data)

  # Predecimos usando Prophet
  df_train = data[['Date','Close']]
  df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

  m = Prophet()
  m.fit(df_train)
  future = m.make_future_dataframe(periods=period)
  forecast = m.predict(future)

  # Mostramos y graficamos el Forecast
  st.subheader('Datos del Forecast')
  st.write(forecast.tail())
      
  st.write(f'Gráfico del Forecast para {n_years} años')
  fig1 = plot_plotly(m, forecast)
  st.plotly_chart(fig1)

  st.write("Componentes del Forecast")
  fig2 = m.plot_components(forecast)
  st.write(fig2)