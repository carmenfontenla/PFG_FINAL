import pandas as pd
from flask import Flask, render_template, request
import pickle
import datetime
from datetime import datetime
#Crear el servidor. Se va a hacer en localhost
app = Flask(__name__)
# 'waitress-serve --listen=0.0.0.0:8000 PaginaWeb:app' hay que poner esto en la terminal para poder acceder a la pagina web
# http://localhost:8000/Predicciones hay que poner eso en el buscador de google para poder ver lo que se hace en la pagina

@app.route('/')
def index():
    return render_template('index.html')

def predicciones(input):
    with open('MODELOS/model_promediodiario.pkl', 'rb') as file:
        model_promediodiario = pickle.load(file)
    print('Calculando la predicción')
    ultima_fecha = datetime.strptime('01/12/2019', '%d/%m/%Y').date()
    fecha_viaje = input[0]
    fecha_viaje_correcto = datetime.strptime(fecha_viaje, '%d/%m/%Y').date()
    diferencia_meses = fecha_viaje_correcto.month - ultima_fecha.month
    df_future = model_promediodiario.make_future_dataframe(periods = diferencia_meses + 1, freq = 'MS')
    prediccion = model_promediodiario.predict(df_future)
    final_predictions = prediccion.iloc[-1, [2, 3]]
    return final_predictions

@app.route('/resultado', methods = ['POST'])
def resultado():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        try:
            resultado = predicciones(predict_list)
            print(f'El promedio diario de vehículos el día {predict_list} es: {resultado}.')
        except ValueError:
            print('Error del sistema. Intentelo de nuevo')
    return render_template('resultados.html', prediction = resultado)

if __name__ == '__main__':
    app.run(port = 5001, debug = True)