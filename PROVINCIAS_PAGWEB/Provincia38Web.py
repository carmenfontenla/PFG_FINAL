import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from scipy.stats import poisson
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from tabulate import tabulate
import pmdarima as pm
from dateutil.relativedelta import relativedelta
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_csv('BBDD/Datos_preprocesados_accidentes_y_victimas_accidentes.csv')

db_provincia38 = df[df['COD_PROVINCIA'] == 38]

num_victimas_dia_semana38 = db_provincia38.groupby(['ANYO', 'MES', 'DIA_SEMANA'])['TOTAL_VICTIMAS_24H'].sum()

num_victimas_dia_semana38 = num_victimas_dia_semana38.reset_index()

num_victimas_dia_semana38['FECHA'] = num_victimas_dia_semana38.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_accidentes_por_dia_semana38 = db_provincia38.groupby(['ANYO', 'MES'])['DIA_SEMANA'].value_counts()

num_accidentes_por_dia_semana38 = num_accidentes_por_dia_semana38.reset_index()

num_accidentes_por_dia_semana38['FECHA'] = num_accidentes_por_dia_semana38.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_victimas_dia_semana38 = num_victimas_dia_semana38.reset_index()

num_victimas_dia_semana38['FECHA'] = num_victimas_dia_semana38.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

df_provincia38 = num_accidentes_por_dia_semana38.merge(num_victimas_dia_semana38, on = 'FECHA')

df_provincia38 = df_provincia38[['ANYO_x', 'FECHA', 'count', 'TOTAL_VICTIMAS_24H']]

df_provincia38.sort_values(by='FECHA')

df_provincia38.to_csv('./BBDD/Provincia38.csv')

# ACCIDENTES

df_provincia38['y'] = df_provincia38['count']
df_provincia38['ds'] = pd.to_datetime(pd.to_datetime(df_provincia38['FECHA']).dt.date)

ts = df_provincia38[['ds', 'y']]
model_a = Prophet(
   yearly_seasonality=True,
   seasonality_mode=['additive','multiplicative'][0]
   ).add_country_holidays(country_name='ESP'
   ).fit(ts)

with open('MODELOS/model_Provincia38_Accidentes.pkl', 'wb') as file:
    pickle.dump(model_a, file)  

# VÍCTIMAS

df_provincia38['y'] = df_provincia38['TOTAL_VICTIMAS_24H']
df_provincia38['ds'] = pd.to_datetime(pd.to_datetime(df_provincia38['FECHA']).dt.date)

ts = df_provincia38[['ds', 'y']]
model_v = Prophet(
   yearly_seasonality=True,
   seasonality_mode=['additive','multiplicative'][0]
   ).add_country_holidays(country_name='ESP'
   ).fit(ts)

with open('MODELOS/model_Provincia38_Victimas.pkl', 'wb') as file:
    pickle.dump(model_v, file)  