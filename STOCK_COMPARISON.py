from matplotlib import ticker
import streamlit as st
import yfinance as yf
import pandas as pd
from streamlit_tags import st_tags, st_tags_sidebar

def relativeret(df):
  rel = df.pct_change()
  cumret = (1+rel).cumprod() - 1
  cumret = cumret.fillna(0)
  return cumret

def main():

  st.header("Comparacion de Acciones")

  dropdown = st_tags(
    label='### Acciones a comparar:',
    text='Presiona enter para agregar la acción',
    suggestions=['TSLA', 'BVN', 'AAPL', 'GOOGL', 'AMZN']
  )

  start = st.date_input('Inicio', value = pd.to_datetime('2021-01-01'))
  end = st.date_input('Final', value = pd.to_datetime('today'))

  if len(dropdown) > 0:
    df = relativeret(
      yf.download(dropdown, start, end)['Adj Close']
    )
    st.line_chart(df)

# main()