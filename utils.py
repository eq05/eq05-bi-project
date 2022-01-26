from datetime import date
import streamlit as st
from pandas_datareader import DataReader

TODAY = date.today().strftime("%Y-%m-%d")

@st.cache(allow_output_mutation=True)
def yahoof_load_data(accion, fecha_inicio, fecha_fin):
  dataframe = DataReader(
      accion, 
      data_source='yahoo', 
      start=fecha_inicio, 
      end=fecha_fin
  )
  return dataframe