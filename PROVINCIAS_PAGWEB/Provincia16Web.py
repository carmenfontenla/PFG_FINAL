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

df = pd.read_csv('BBDD/Datos_preprocesados_accidentes_y_victimas_accidentes.csv')

db_provincia16 = df[df['COD_PROVINCIA'] == 16]

num_victimas_dia_semana16 = db_provincia16.groupby(['ANYO', 'MES', 'DIA_SEMANA'])['TOTAL_VICTIMAS_24H'].sum()

num_victimas_dia_semana16 = num_victimas_dia_semana16.reset_index()

num_victimas_dia_semana16['FECHA'] = num_victimas_dia_semana16.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_accidentes_por_dia_semana16 = db_provincia16.groupby(['ANYO', 'MES'])['DIA_SEMANA'].value_counts()

num_accidentes_por_dia_semana16 = num_accidentes_por_dia_semana16.reset_index()

num_accidentes_por_dia_semana16['FECHA'] = num_accidentes_por_dia_semana16.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_victimas_dia_semana16 = num_victimas_dia_semana16.reset_index()

num_victimas_dia_semana16['FECHA'] = num_victimas_dia_semana16.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

df_provincia16 = num_accidentes_por_dia_semana16.merge(num_victimas_dia_semana16, on = 'FECHA')

df_provincia16 = df_provincia16[['ANYO_x', 'FECHA', 'count', 'TOTAL_VICTIMAS_24H']]

df_provincia16.sort_values(by='FECHA')

df_provincia16.to_csv('./BBDD/Provincia16.csv')

# ACCIDENTES NINGUNO

# VÍCTIMAS NINGUNO
