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
import statsmodels as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
import pmdarima as pm
from dateutil.relativedelta import relativedelta
import pickle

#vamos a usar prophet dado que fue el modelo con mejores resultados (error <15%)
promedio_diario = pd.read_csv('BBDD/evolucion_del_trafico_medio_en_las_autopistas_estatales_de_peaje_en_españa.csv', sep = ';')

promedio_diario.drop(promedio_diario[promedio_diario['Año'] == '2015'].index, inplace = True)

promedio_diario.dropna(inplace = True)

inicio = datetime(2016,1,1)

lista_fechas = [inicio + relativedelta(months=+d) for d in range(0,48)]
print(lista_fechas)

promedio_diario['FECHA'] = lista_fechas

promedio_diario['y'] = promedio_diario['Intensidad media diaria de vehículos']
promedio_diario['ds'] = pd.to_datetime(pd.to_datetime(promedio_diario['FECHA']).dt.date)
promedio_diario.to_csv('./BBDD/Promedio_Diario.csv')

ts = promedio_diario[['ds', 'y']]
model = Prophet(
   yearly_seasonality=True,
   seasonality_mode=['additive','multiplicative'][0]
   ).add_country_holidays(country_name='ESP'
   ).fit(ts)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

from prophet.diagnostics import cross_validation
cortes = pd.date_range(start = '2018-01-01', end = '2019-09-01', freq = 'MS')
df_cv = cross_validation(model, horizon = '90 days', cutoffs = cortes)

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)

with open('MODELOS/model_promediodiario.pkl', 'wb') as file:
    pickle.dump(model, file)

