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

db_provincia20 = df[df['COD_PROVINCIA'] == 20]

num_victimas_dia_semana20 = db_provincia20.groupby(['ANYO', 'MES', 'DIA_SEMANA'])['TOTAL_VICTIMAS_24H'].sum()

num_victimas_dia_semana20 = num_victimas_dia_semana20.reset_index()

num_victimas_dia_semana20['FECHA'] = num_victimas_dia_semana20.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_accidentes_por_dia_semana20 = db_provincia20.groupby(['ANYO', 'MES'])['DIA_SEMANA'].value_counts()

num_accidentes_por_dia_semana20 = num_accidentes_por_dia_semana20.reset_index()

num_accidentes_por_dia_semana20['FECHA'] = num_accidentes_por_dia_semana20.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_victimas_dia_semana20 = num_victimas_dia_semana20.reset_index()

num_victimas_dia_semana20['FECHA'] = num_victimas_dia_semana20.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

df_provincia20 = num_accidentes_por_dia_semana20.merge(num_victimas_dia_semana20, on = 'FECHA')

df_provincia20 = df_provincia20[['ANYO_x', 'FECHA', 'count', 'TOTAL_VICTIMAS_24H']]

# ACCIDENTES NINGUNO

# VÍCTIMAS

df_provincia20['y'] = df_provincia20['TOTAL_VICTIMAS_24H']
df_provincia20['ds'] = pd.to_datetime(pd.to_datetime(df_provincia20['FECHA']).dt.date)

ts = df_provincia20[['ds', 'y']]
model_v = Prophet(
   yearly_seasonality=True,
   seasonality_mode=['additive','multiplicative'][0]
   ).add_country_holidays(country_name='ESP'
   ).fit(ts)

with open('MODELOS/model_Provincia20_Victimas.pkl', 'wb') as file:
    pickle.dump(model_v, file)  