import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten

def diferencia_meses(fecha_viaje, ultima_fecha_provincia):
    ultima_fecha_promedio_diario = '01/01/2019'
    ultima_fecha_correcta_promedio_diario = datetime.strptime(ultima_fecha_promedio_diario, '%d/%m/%Y').date()
    fecha_viaje_correcto = datetime.strptime(fecha_viaje, '%d/%m/%Y').date()
    ultima_fecha_correcta_provincia = datetime.strptime(ultima_fecha_provincia, '%d/%m/%Y').date()
    diferencia_promedio_diario = relativedelta(fecha_viaje_correcto, ultima_fecha_correcta_promedio_diario)
    diferencia_provincia = fecha_viaje_correcto - ultima_fecha_correcta_provincia
    diferencia_dias = diferencia_provincia.days
    diferencia_meses_promedio_diario = diferencia_promedio_diario.years*12 + diferencia_promedio_diario.months
    print(diferencia_meses_promedio_diario)
    return diferencia_meses_promedio_diario, diferencia_dias


def predicciones_promedio_diario(diferencia_meses_promedio_diario):
    with open('./MODELOS/model_promediodiario.pkl', 'rb') as file:
        model_promediodiario = pickle.load(file)
    if diferencia_meses_promedio_diario <= 0:
        df = pd.read_csv('./BBDD/Promedio_Diario.csv')
        prediccion_redondeada_promedio_diario = df.loc[df.index[-1 + diferencia_meses], 'y']
    else:
        df_future = model_promediodiario.make_future_dataframe(periods = diferencia_meses_promedio_diario + 1, freq = 'MS')
        prediccion = model_promediodiario.predict(df_future)
        prediccion_final_promedio_diario = prediccion.iloc[-1]['yhat_upper']
        prediccion_redondeada_promedio_diario = np.round(prediccion_final_promedio_diario)
    return prediccion_redondeada_promedio_diario


def predicciones_prophet(diferencia_dias, nombre_modelo, database, variable, fecha_viaje):
    with open('./MODELOS/' + nombre_modelo, 'rb') as file:
        model_provincia = pickle.load(file)
    if diferencia_dias <= 0:
        df = pd.read_csv('./BBDD/' + database)
        dia, mes, anyo = fecha_viaje.split('/')
        fecha_busqueda = f'{anyo}/{mes}/{dia}'
        prediccion_redondeada_prophet = df[df['FECHA'] == fecha_busqueda][variable].values[0]
    else:
        df_future = model_provincia.make_future_dataframe(periods = diferencia_dias + 1, freq = 'D')
        prediccion = model_provincia.predict(df_future)
        prediccion_final_provincia = prediccion.iloc[-1]['yhat_upper']
        prediccion_redondeada_prophet = np.round(prediccion_final_provincia)
    return prediccion_redondeada_prophet

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


def crear_modeloFF():
    model = Sequential()
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse',optimizer='Adam',metrics=["mae"])
    model.summary()
    return model


def prediccion_redes_neuronales(diferencia_dias, nombre_modelo, dataset, archivo_scaler, variable, fecha_viaje):
    df_provincia = pd.read_csv('./BBDD/' + dataset)
    with open('./MODELOS/' + archivo_scaler ,'rb') as f:
        scaler = pickle.load(f)
    if diferencia_dias <= 0:
        dia, mes, anyo = fecha_viaje.split('/')
        fecha_busqueda = f'{anyo}/{mes}/{dia}'
        prediccion_redondeada_redes_neuronales = df_provincia[df_provincia['FECHA'] == fecha_busqueda][variable].values[0]
    else:
        model = crear_modeloFF()
        model.save_weights('./MODELOS/' + nombre_modelo)
        ultimos_datos = df_provincia[: -PASOS*2]['count'].values
        ultimos_valores = ultimos_datos.reshape(-1, 1)
        datos_escalados = scaler.transform(ultimos_valores)
        datos_supervisados = series_to_supervised(datos_escalados, PASOS, 1)
        datos_supervisados.drop(datos_supervisados.columns[[PASOS]], axis = 1, inplace = True)
        valores_pre = datos_supervisados.values
        x_test = valores_pre[PASOS - 1:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        def agregarNuevoValor(x_test,nuevoValor):
            for i in range(x_test.shape[2]-1):
                x_test[0][0][i] = x_test[0][0][i+1]
            x_test[0][0][x_test.shape[2]-1]=nuevoValor
            return x_test
        results=[]
        for i in range(diferencia_dias):
            parcial=model.predict(x_test)
            results.append(parcial[0])
            x_test=agregarNuevoValor(x_test,parcial[0])
        results = [x for x in results]
        datos_predichos = scaler.inverse_transform(results)
        prediccion_redondeada_redes_neuronales = np.round(datos_predichos[-1])
    return prediccion_redondeada_redes_neuronales