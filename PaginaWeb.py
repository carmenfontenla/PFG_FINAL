import pandas as pd
from flask import Flask, jsonify, request
import pickle
import datetime
from datetime import datetime
#Crear el servidor. Se va a hacer en localhost
app = Flask(__name__)
@app.route('/Predicciones', methods = ['POST'])
# 'waitress-serve --listen=0.0.0.0:8000 PaginaWeb:app' hay que poner esto en la terminal para poder acceder a la pagina web
# http://localhost:8000/Predicciones hay que poner eso en el buscador de google para poder ver lo que se hace en la pagina
def predicciones():
    try:
        req_json = request.get_json()
        input = pd.read_json(req_json, orient = 'records')
    except Exception as e:
        raise e
    if input.empty:
        return (bad_request())
    else:
        with open('MODELOS/model_promediodiario.pkl', 'rb') as file:
            model_promediodiario = pickle.dump(file)
        print('Calculando la predicción')
        ultima_fecha = datetime.strptime('01/12/2019', '%d/%m/%Y').date()
        fecha_viaje = input.iloc[0, 0]
        fecha_viaje_correcto = datetime.strptime(fecha_viaje, '%d/%m/%Y').date()
        diferencia_meses = fecha_viaje_correcto.month - ultima_fecha.month
        df_future = model_promediodiario.make_future_dataframe(periods = diferencia_meses + 1, freq = 'MS')
        prediccion = model_promediodiario.predict(df_future)
        final_predictions = prediccion.iloc[-1, [2, 3]]
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        return responses