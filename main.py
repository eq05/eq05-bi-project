import streamlit as st

from LSTM import main as LSTM_main
from PROPHET import main as PROPHET_main
from SVC import main as SVC_main
from LINEAR_REGRESION import main as LINEAR_REGRESION_main
from STOCK_COMPARISON import main as STOCK_COMPARISON_main
from RANDOM_FOREST import main as RANDOM_FOREST_main

st.sidebar.title("Proyecto BI - Eq05")

with st.sidebar.expander("Información", True):
  st.markdown("""

# Sistema Web de Inteligencia de Negocios para soporte a la Toma de Decisiones de Inversiones en Bolsa de Valores
## Curso  
**Inteligencia de Negocios**

## Profesor: 
**Cancho Rodríguez, Ernesto**

## Escuela Profesional:  
**Ingeniería de Software** 

## Grupo 5

* **Linares Purizaca, Mauricio Javier** (18200086)
* **Marcos de la Torre, Renzo Alexis** (18200274)
* **Mejía Tarazona, Brandon Isaac** (18200276)
* **Oroncuy Fernandez, Brayan Richard** (18200282)
* **Ramos Paredes, Roger Anthony** (18200096)
* **Salcedo Alfaron, Jhon Marco** (18200101)
* **Vargas Pizango, Sebastian Enrique** (18200104)

""")

with st.expander("Ejecuta un modelo", True):

  st.title("Seleccione el modelo a correr")

  app = st.radio(
      "Modelos disponibles:",
      ('Ninguno','Facebook Prophet','LSTM', 'SVC', 'Regresión Lineal', 'Random Forest', 'Comparación de acciones'))

with st.container():
  if  app == 'Ninguno':
    st.info('Seleccione un modelo')
  elif  app == 'Facebook Prophet':
    st.title("Facebook Prophet.")
    PROPHET_main()
  elif app == 'LSTM':
    st.title("LSTM.")
    LSTM_main()
  elif app == 'SVC':
    st.title("SVC.")
    SVC_main()
  elif app == 'Regresión Lineal':
    LINEAR_REGRESION_main()
  elif app == 'Comparación de acciones':
    STOCK_COMPARISON_main()
  elif app == 'Random Forest':
    RANDOM_FOREST_main()
  
