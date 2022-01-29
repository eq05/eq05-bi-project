import os
import sys
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import streamlit as st
import yfinance as yf
import pandas as pd
from streamlit_tags import st_tags, st_tags_sidebar
from functools import reduce
import datetime

def get_merge_df(tickers, start, end):
  dataframes = []
  for ticker in tickers:
    df_yf = yf.download(ticker, start, end)
    df_yf['Empresa'] = ticker
    df_yf["Date"] = df_yf.index
    df_yf = df_yf.reset_index(level=0, drop=True).reset_index()
    df_yf.drop("index", axis=1, inplace=True)
    dataframes.append(df_yf)
  df_merged = reduce(lambda left, right: pd.merge(left, right, how='outer'), dataframes)
  df_merged['Date'] = df_merged['Date'].apply(lambda d: d.strftime("%Y-%m-%d"))
  return df_merged

def get_new_key():
  for key in [i for i in range(10000)]:
    yield str(key)

def obv(group):

  volume = group['Volume']
  change = group['Close'].diff()

  prev_obv = 0
  obv_values = []

  for i, j in zip(change, volume):

      if i > 0:
          current_obv = prev_obv + j
      elif i < 0:
          current_obv = prev_obv - j
      else:
          current_obv = prev_obv

      prev_obv = current_obv
      obv_values.append(current_obv)
  
  return pd.Series(obv_values, index = group.index)

def show_refs():
  st.subheader('Referencias')
  st.markdown("""
**Eduardo Navarro** | Eq2 - Modelo Random Forest ([Ver video](https://www.youtube.com/watch?v=q5n639HXEZA))
""")

def main():
  keys_generator = get_new_key()

  st.header("Random Forest")

  dropdown = st_tags(
    label='### Acciones a cargar:',
    text='Presiona enter para agregar la acción',
    suggestions=['TSLA', 'BVN', 'AAPL', 'GOOGL', 'AMZN', 'FB', 'MSFT', 'NVDA', 'SHOP']
  )

  tickers = [t.upper() for t in dropdown]
  
  start = st.date_input('Inicio', value = pd.to_datetime('2021-01-01'))
  end = st.date_input('Final', value = pd.to_datetime('today'))


  if len(tickers) >= 2: 
    df = get_merge_df(tickers, start, end)
    cant_empresas = len(df['Empresa'].value_counts().index)
    if (cant_empresas < 2):
      st.warning("Ingrese al menos dos acciones validas")
      show_refs()
      return
  else: 
    st.warning("Ingrese al menos dos acciones")
    show_refs()
    return

  st.subheader("Dataframe cargado:")
  st.write(df)

  price_data = df

  st.subheader("Filtrado de columnas")
  price_data = price_data[['Empresa','Date','Close','High','Low','Open','Volume']]
  price_data.sort_values(by = ['Empresa','Date'], inplace = True)
  price_data['change_in_price'] = price_data['Close'].diff()

  mask = price_data['Empresa'] != price_data['Empresa'].shift(1)
  price_data['change_in_price'] = np.where(mask == True, np.nan, price_data['change_in_price'])
  st.write(price_data[price_data.isna().any(axis = 1)])

  with st.expander("Preprocesamiento", False):

    st.subheader("Preprocesamiento: Smoothing the Data")
    days_out = st.number_input("days_out", 1, 30, 30, key=next(keys_generator))
    price_data_smoothed = price_data.groupby(['Empresa'])[['Close','Low','High','Open','Volume']].transform(lambda x: x.ewm(span = days_out).mean())
    smoothed_df = pd.concat([price_data[['Empresa','Date']], price_data_smoothed], axis=1, sort=False)
    st.write(smoothed_df)

    st.subheader("Preprocesamiento: Signal Flag")
    days_out = st.number_input("days_out", 1, 30, 30, key=next(keys_generator))
    smoothed_df['Signal_Flag'] = smoothed_df.groupby('Empresa')['Close'].transform(lambda x : np.sign(x.diff(days_out)))
    st.write(smoothed_df.head(50))


  with st.expander("Cálculo de indicadores", False):

    st.subheader("Cálculo del indicador: índice de fuerza relativa (RSI)")
    n = st.number_input("n", 1, 30, 14, key=next(keys_generator))
    up_df, down_df = price_data[['Empresa','change_in_price']].copy(), price_data[['Empresa','change_in_price']].copy()
    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
    down_df['change_in_price'] = down_df['change_in_price'].abs()
    ewma_up = up_df.groupby('Empresa')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('Empresa')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    relative_strength = ewma_up / ewma_down
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))
    price_data['down_days'] = down_df['change_in_price']
    price_data['up_days'] = up_df['change_in_price']
    price_data['RSI'] = relative_strength_index
    st.write(price_data.head(30))

    st.subheader("Cálculo del indicador: oscilador estocástico (Stochastic Oscillator)")
    n = st.number_input("n", 1, 30, 14, key=next(keys_generator))
    low_14, high_14 = price_data[['Empresa','Low']].copy(), price_data[['Empresa','High']].copy()
    low_14 = low_14.groupby('Empresa')['Low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('Empresa')['High'].transform(lambda x: x.rolling(window = n).max())
    k_percent = 100 * ((price_data['Close'] - low_14) / (high_14 - low_14))
    price_data['low_14'] = low_14
    price_data['high_14'] = high_14
    price_data['k_percent'] = k_percent
    st.write(price_data.head(30))

    st.subheader('Cálculo del indicador: Williams %R')
    n = 14
    low_14, high_14 = price_data[['Empresa','Low']].copy(), price_data[['Empresa','High']].copy()
    low_14 = low_14.groupby('Empresa')['Low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('Empresa')['High'].transform(lambda x: x.rolling(window = n).max())
    r_percent = ((high_14 - price_data['Close']) / (high_14 - low_14)) * - 100
    price_data['r_percent'] = r_percent
    st.write(price_data.head(30))

    st.subheader('Cálculo del indicador: media móvil convergencia divergencia (MACD)')
    ema_26 = price_data.groupby('Empresa')['Close'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = price_data.groupby('Empresa')['Close'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26
    ema_9_macd = macd.ewm(span = 9).mean()
    price_data['MACD'] = macd
    price_data['MACD_EMA'] = ema_9_macd
    st.write(price_data.head(30))

    st.subheader('Cálculo del indicador: tasa de cambio de precio')
    n = st.number_input("n", 1, 30, 9, key=next(keys_generator))
    price_data['Price_Rate_Of_Change'] = price_data.groupby('Empresa')['Close'].transform(lambda x: x.pct_change(periods = n))
    st.write(price_data.head(30))

    st.subheader('Cálculo del indicador: Volumen en equilibrio')
    obv_groups = price_data.groupby('Empresa').apply(obv)
    price_data['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)
    st.write(price_data.head(30))


  with st.expander("Construcción del modelo", False):

    st.subheader('Construcción del modelo: creación de la columna de predicción')
    close_groups = price_data.groupby('Empresa')['Close']
    close_groups = close_groups.transform(lambda x : np.sign(x.diff()))
    price_data['Prediction'] = close_groups
    price_data.loc[price_data['Prediction'] == 0.0] = 1.0
    st.write(price_data.head(50))

    st.subheader('Construcción del modelo: Eliminación de valores de NaN')
    st.info('Antes de eliminar filas NaN se tiene {} filas y {} columnas'.format(price_data.shape[0], price_data.shape[1]))
    price_data = price_data.dropna()
    st.info('Después de eliminar filas NaN se tiene {} filas y {} columnas'.format(price_data.shape[0], price_data.shape[1]))
    st.write(price_data.head())

    st.subheader('Construcción del modelo: división de los datos')
    st.code("""
  X_Cols = price_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
  Y_Cols = price_data['Prediction']

  X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

  rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

  rand_frst_clf.fit(X_train, y_train)

  y_pred = rand_frst_clf.predict(X_test)
    """)
    X_Cols = price_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
    Y_Cols = price_data['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)
    rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)
    rand_frst_clf.fit(X_train, y_train)
    y_pred = rand_frst_clf.predict(X_test)

  with st.expander("Evaluación del modelo", False):

    st.subheader('Evaluación del modelo: precisión')
    st.info(f'Correct Prediction (%): {accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0}')

    st.subheader('Modelo de Evaluación: Informe de Clasificación')
    target_names = ['Down Day', 'Up Day']
    report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    st.subheader('Evaluación del modelo: matriz de confusión')

    rf_matrix = confusion_matrix(y_test, y_pred)

    true_negatives = rf_matrix[0][0]
    false_negatives = rf_matrix[1][0]
    true_positives = rf_matrix[1][1]
    false_positives = rf_matrix[0][1]

    accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    percision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    st.info('Accuracy: {}'.format(float(accuracy)))
    st.info('Percision: {}'.format(float(percision)))
    st.info('Recall: {}'.format(float(recall)))
    st.info('Specificity: {}'.format(float(specificity)))

    disp = plot_confusion_matrix(rand_frst_clf, X_test, y_test, display_labels = ['Down Day', 'Up Day'], normalize = 'true', cmap=plt.cm.Blues)
    disp.ax_.set_title('Matriz de confusión')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.subheader('Evaluación del modelo: importancia de las características')
    feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)
    st.write(feature_imp)

    st.subheader('Evaluación del modelo: representación gráfica de la importancia de las características')
    x_values = list(range(len(rand_frst_clf.feature_importances_)))
    cumulative_importances = np.cumsum(feature_imp.values)
    plt.plot(x_values, cumulative_importances, 'g-')
    plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')
    plt.xticks(x_values, feature_imp.index, rotation = 'vertical')
    plt.xlabel('Variable')
    plt.ylabel('Importancia acumulada')
    plt.title('Random Forest: Gráfico de importancia de características')
    st.pyplot()

    st.subheader('Evaluación del modelo: Curva ROC')
    rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8)
    st.pyplot()

  st.subheader('Evaluación del modelo: puntaje de error fuera de la bolsa')
  st.success('Random Forest Out-Of-Bag Error Score: {}'.format(rand_frst_clf.oob_score_))

  show_refs()
# main()