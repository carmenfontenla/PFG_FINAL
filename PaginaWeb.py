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
    final_predictions = prediccion.iloc[-1, [2]]
    return final_predictions

modelos_provincias_accidentes = {
    1:('model_Provincia1_Accidentes.pkl', 'PROPHET', 0, 0),
    2:('model_Provincia2_Accidentes.pkl', 'RN', 'scaler_Provincia2_accidentes.sav', 'accidentes_Provincia2.weights.h5'),
    3:('model_Provincia3_Accidentes.pkl', 'PROPHET', 0, 0),
    4:('model_Provincia4_Accidentes.pkl', 'PROPHET', 0, 0),
    5:(0, 'No existe', 0, 0),
    6:(0, 'No existe', 0, 0),
    7:('model_Provincia7_Accidentes.pkl', 'PROPHET', 0, 0),
    8:(0, 'No existe', 0, 0),
    9:(0, 'No existe', 0, 0),
    10:(0, 'No existe', 0, 0),
    11:('model_Provincia11_Accidentes.pkl', 'PROPHET', 0, 0),
    12:(0, 'No existe', 0, 0),
    13:(0, 'No existe', 0, 0),
    14:('model_Provincia14_Accidentes.pkl', 'RN', 'scaler_Provincia14_accidentes.sav', 'accidentes_Provincia14.weights.h5'),
    15:('model_Provincia15_Accidentes.pkl', 'PROPHET', 0, 0),
    16:(0, 'No existe', 0, 0),
    17:('model_Provincia17_Accidentes.pkl', 'PROPHET', 0, 0),
    18:('model_Provincia18_Accidentes.pkl', 'RN', 'scaler_Provincia18_accidentes.sav', 'accidentes_Provincia18.weights.h5'),
    19:(0, 'No existe', 0, 0),
    20:(0, 'No existe', 0, 0),
    21:(0, 'No existe', 0, 0),
    22:(0, 'No existe', 0, 0),
    23:('model_Provincia23_Accidentes.pkl', 'PROPHET', 0, 0),
    24:(0, 'No existe', 0, 0),
    25:('model_Provincia25_Accidentes.pkl', 'RN', 'scaler_Provincia25_accidentes.sav', 'accidentes_Provincia25.weights.h5'),
    26:('model_Provincia26_Accidentes.pkl', 'PROPHET', 0, 0),
    27:(0, 'No existe', 0, 0),    
    28:('model_Provincia28_Accidentes.pkl', 'PROPHET', 0, 0),
    29:(0, 'No existe', 0, 0),
    30:('model_Provincia30_Accidentes.pkl', 'PROPHET', 0, 0),
    31:(0, 'No existe', 0, 0),
    32:(0, 'No existe', 0, 0),    
    33:('model_Provincia33_Accidentes.pkl', 'PROPHET', 0, 0),
    34:(0, 'No existe', 0, 0),    
    35:('model_Provincia35_Accidentes.pkl', 'PROPHET', 0, 0),
    36:('model_Provincia36_Accidentes.pkl', 'PROPHET', 0, 0),
    37:(0, 'No existe', 0, 0),    
    38:('model_Provincia38_Accidentes.pkl', 'PROPHET', 0, 0),
    39:('model_Provincia39_Accidentes.pkl', 'RN', 'scaler_Provincia39_accidentes.sav', 'accidentes_Provincia39.weights.h5'),
    40:(0, 'No existe', 0, 0),    
    41:('model_Provincia41_Accidentes.pkl', 'PROPHET', 0, 0),
    42:(0, 'No existe', 0, 0),    
    43:('model_Provincia43_Accidentes.pkl', 'PROPHET', 0, 0),
    44:(0, 'No existe', 0, 0),    
    45:('model_Provincia45_Accidentes.pkl', 'PROPHET', 0, 0),
    46:('model_Provincia46_Accidentes.pkl', 'PROPHET', 0, 0),
    47:(0, 'No existe', 0, 0),
    48:(0, 'No existe', 0, 0), 
    49:('model_Provincia49_Accidentes.pkl', 'RN', 'scaler_Provincia49_accidentes.sav', 'accidentes_Provincia49.weights.h5'),
    50:('model_Provincia50_Accidentes.pkl', 'PROPHET', 0, 0),
    51:(0, 'No existe', 0, 0),
    52:(0, 'No existe', 0, 0), 
}

modelos_provincias_victimas = {
    1:('model_Provincia1_Accidentes.pkl', 'PROPHET', 0, 0),
    2:('model_Provincia2_Accidentes.pkl', 'PROPHET', 0, 0),
    3:('model_Provincia3_Accidentes.pkl', 'PROPHET', 0, 0),
    4:(0, 'No existe', 0, 0),
    5:(0, 'No existe', 0, 0),
    6:(0, 'No existe', 0, 0),
    7:('model_Provincia7_Accidentes.pkl', 'PROPHET', 0, 0),
    8:('model_Provincia8_Accidentes.pkl', 'RN', 'scaler_Provincia8_accidentes.sav', 'accidentes_Provincia8.weights.h5'),
    9:(0, 'No existe', 0, 0),
    10:('model_Provincia10_Accidentes.pkl', 'RN', 'scaler_Provincia10_accidentes.sav', 'accidentes_Provincia10.weights.h5'),
    11:('model_Provincia11_Accidentes.pkl', 'PROPHET', 0, 0),
    12:(0, 'No existe', 0, 0),
    13:(0, 'No existe', 0, 0),
    14:(0, 'No existe', 0, 0),
    15:('model_Provincia15_Accidentes.pkl', 'PROPHET', 0, 0),
    16:(0, 'No existe', 0, 0),
    17:('model_Provincia17_Accidentes.pkl', 'PROPHET', 0, 0),
    18:('model_Provincia18_Accidentes.pkl', 'PROPHET', 0, 0),
    19:(0, 'No existe', 0, 0),
    20:('model_Provincia20_Accidentes.pkl', 'PROPHET', 0, 0),
    21:('model_Provincia21_Accidentes.pkl', 'RN', 'scaler_Provincia21_accidentes.sav', 'accidentes_Provincia21.weights.h5'),
    22:(0, 'No existe', 0, 0),
    23:(0, 'No existe', 0, 0),
    24:(0, 'No existe', 0, 0),
    25:(0, 'No existe', 0, 0),
    26:(0, 'No existe', 0, 0),
    27:(0, 'No existe', 0, 0),    
    28:('model_Provincia28_Accidentes.pkl', 'PROPHET', 0, 0),
    29:('model_Provincia29_Accidentes.pkl', 'RN', 'scaler_Provincia29_accidentes.sav', 'accidentes_Provincia29.weights.h5'),
    30:('model_Provincia30_Accidentes.pkl', 'PROPHET', 0, 0),
    31:('model_Provincia31_Accidentes.pkl', 'RN', 'scaler_Provincia31_accidentes.sav', 'accidentes_Provincia31.weights.h5'),
    32:(0, 'No existe', 0, 0),    
    33:('model_Provincia33_Accidentes.pkl', 'PROPHET', 0, 0),
    34:(0, 'No existe', 0, 0),    
    35:('model_Provincia35_Accidentes.pkl', 'PROPHET', 0, 0),
    36:(0, 'No existe', 0, 0),    
    37:(0, 'No existe', 0, 0),    
    38:('model_Provincia38_Accidentes.pkl', 'PROPHET', 0, 0),
    39:(0, 'No existe', 0, 0),    
    40:(0, 'No existe', 0, 0),    
    41:('model_Provincia41_Accidentes.pkl', 'PROPHET', 0, 0),
    42:(0, 'No existe', 0, 0),  
    43:('model_Provincia43_Accidentes.pkl', 'PROPHET', 0, 0),  
    44:(0, 'No existe', 0, 0),    
    45:(0, 'No existe', 0, 0),    
    46:('model_Provincia46_Accidentes.pkl', 'RN', 'scaler_Provincia46_accidentes.sav', 'accidentes_Provincia46.weights.h5'),
    47:('model_Provincia47_Accidentes.pkl', 'RN', 'scaler_Provincia47_accidentes.sav', 'accidentes_Provincia47.weights.h5'),
    48:('model_Provincia48_Accidentes.pkl', 'RN', 'scaler_Provincia48_accidentes.sav', 'accidentes_Provincia48.weights.h5'),
    49:(0, 'No existe', 0, 0),
    50:(0, 'No existe', 0, 0), 
    51:(0, 'No existe', 0, 0),
    52:(0, 'No existe', 0, 0),
}

@app.route('/resultado', methods = ['POST'])
def resultado():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        try:
            resultado = predicciones(predict_list)
        except ValueError:
            resultado = 'Error del sistema. Intentelo de nuevo'
    return render_template('resultados.html', prediction = resultado)

if __name__ == '__main__':
    app.run(port = 5001, debug = True)
