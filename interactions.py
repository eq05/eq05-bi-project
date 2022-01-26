import streamlit as st
from datetime import date
from utils import *
import numpy as np
import pandas as pd

def select_dataset():
  st.text("Selección del dataset")
  
  selected_stock = st.text_input('Ingresa la acción', "AAPL")
  date_ini, date_fin = select_range_date()

  try:
    data_load_state = st.text('Cargando datos...')
    dataframe = yahoof_load_data(selected_stock, date_ini, date_fin)
    data_load_state.text('Cargando datos... ¡Listo!')
    return dataframe, True
  except:
    data_load_state.text('Cargando datos... ¡Error!')
    st.error(f"La acción {selected_stock} no existe o no es un valor valido, por favor ingrese el valor nuevamente.")
    return pd.DataFrame({}), False


def select_range_date():
  date_ini = st.date_input(
     "Fecha inicio",
     date(2019, 1, 1)
  )
  st.write('Fecha selecionada:', date_ini)

  date_fin = st.date_input(
     "Fecha fin",
     date.today()
  )
  st.write('Fecha selecionada:', date_fin)

  return date_ini.strftime("%Y-%m-%d"), date_fin.strftime("%Y-%m-%d")