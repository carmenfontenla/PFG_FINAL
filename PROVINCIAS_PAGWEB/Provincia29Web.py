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

db_provincia29 = df[df['COD_PROVINCIA'] == 29]

num_victimas_dia_semana29 = db_provincia29.groupby(['ANYO', 'MES', 'DIA_SEMANA'])['TOTAL_VICTIMAS_24H'].sum()

num_victimas_dia_semana29 = num_victimas_dia_semana29.reset_index()

num_victimas_dia_semana29['FECHA'] = num_victimas_dia_semana29.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_accidentes_por_dia_semana29 = db_provincia29.groupby(['ANYO', 'MES'])['DIA_SEMANA'].value_counts()

num_accidentes_por_dia_semana29 = num_accidentes_por_dia_semana29.reset_index()

num_accidentes_por_dia_semana29['FECHA'] = num_accidentes_por_dia_semana29.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

num_victimas_dia_semana29 = num_victimas_dia_semana29.reset_index()

num_victimas_dia_semana29['FECHA'] = num_victimas_dia_semana29.apply(lambda row: str(row['ANYO']) + '/' + str(row['MES']) + '/' + str(row['DIA_SEMANA']), axis = 1)

df_provincia29 = num_accidentes_por_dia_semana29.merge(num_victimas_dia_semana29, on = 'FECHA')

df_provincia29 = df_provincia29[['ANYO_x', 'FECHA', 'count', 'TOTAL_VICTIMAS_24H']]

df_provincia29.sort_values(by='FECHA')

df_provincia29.to_csv('./BBDD/Provincia29.csv')

# ACCIDENTES NINGUNO

# VÍCTIMAS

PASOS=84
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
values = df_provincia29['TOTAL_VICTIMAS_24H'].values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.head()

values = reframed.values
n_train_days = 320
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split isnto input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

def crear_modeloFF():
    model_v = Sequential()
    model_v.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model_v.add(Flatten())
    model_v.add(Dense(1, activation='tanh'))
    model_v.compile(loss='mse',optimizer='Adam',metrics=["mae"])
    model_v.summary()
    return model_v

EPOCHS=20

model_v = crear_modeloFF()
 
history=model_v.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)

with open('MODELOS/model_Provincia29_Victimas.pkl', 'wb') as file:
    pickle.dump(model_v, file)

with open('MODELOS/scaler_Provincia29_Victimas.sav', 'wb') as file:
    pickle.dump(scaler, file)

model_v.save_weights('MODELOS/victimas_Provincia29.weights.h5')