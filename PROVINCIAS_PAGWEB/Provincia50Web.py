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

db_provincia50 = df[df['COD_PROVINCIA'] == 50]

num_victimas_dia_semana50 = db_provincia50.groupby(['ANYO', 'MES', 'DIA_SEMANA'])['TOTAL_VICTIMAS_24H'].sum()

num_victimas_dia_semana50 = num_victimas_dia_semana50.reset_index()

num_victimas_dia_semana50['FECHA'] = num_victimas_dia_semana50.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_accidentes_por_dia_semana50 = db_provincia50.groupby(['ANYO', 'MES'])['DIA_SEMANA'].value_counts()

num_accidentes_por_dia_semana50 = num_accidentes_por_dia_semana50.reset_index()

num_accidentes_por_dia_semana50['FECHA'] = num_accidentes_por_dia_semana50.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_victimas_dia_semana50 = num_victimas_dia_semana50.reset_index()

num_victimas_dia_semana50['FECHA'] = num_victimas_dia_semana50.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

df_provincia50 = num_accidentes_por_dia_semana50.merge(num_victimas_dia_semana50, on = 'FECHA')

df_provincia50 = df_provincia50[['ANYO_x', 'FECHA', 'count', 'TOTAL_VICTIMAS_24H']]

# ACCIDENTES

df_provincia50['y'] = df_provincia50['count']
df_provincia50['ds'] = pd.to_datetime(pd.to_datetime(df_provincia50['FECHA']).dt.date)

ts = df_provincia50[['ds', 'y']]
model_a = Prophet(
   yearly_seasonality=True,
   seasonality_mode=['additive','multiplicative'][0]
   ).add_country_holidays(country_name='ESP'
   ).fit(ts)

with open('MODELOS/model_Provincia50_Accidentes.pkl', 'wb') as file:
    pickle.dump(model_a, file)  

# VÍCTIMAS NINGUNO