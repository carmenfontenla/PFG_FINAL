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


modelos_provincias_accidentes = {
    1:('model_Provincia1_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia1.csv'),
    2:('model_Provincia2_Accidentes.pkl', 'RN', 'scaler_Provincia2_accidentes.sav', 'accidentes_Provincia2.weights.h5', '07/12/2020', 'Provincia2.csv'),
    3:('model_Provincia3_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia4.csv'),
    5:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia5.csv'),
    6:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia6.csv'),
    7:('model_Provincia7_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia7.csv'),
    8:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia8.csv'),
    9:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia9.csv'),
    10:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia10.csv'),
    11:('model_Provincia11_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia11.csv'),
    12:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia12.csv'),
    13:(0, 'No existe', 0, 0, '06/12/2020', 'Provincia13.csv'),
    14:('model_Provincia14_Accidentes.pkl', 'RN', 'scaler_Provincia14_accidentes.sav', 'accidentes_Provincia14.weights.h5', '07/12/2020', 'Provincia4.csv'),
    15:('model_Provincia15_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia15.csv'),
    16:(0, 'No existe', 0, 0, '05/12/2020', 'Provincia16.csv'),
    17:('model_Provincia17_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia17.csv'),
    18:('model_Provincia18_Accidentes.pkl', 'RN', 'scaler_Provincia18_accidentes.sav', 'accidentes_Provincia18.weights.h5', '07/12/2020', 'Provincia18.csv'),
    19:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia19.csv'),
    20:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia20.csv'),
    21:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia21.csv'),
    22:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia22.csv'),
    23:('model_Provincia23_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia23.csv'),
    24:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia24.csv'),
    25:('model_Provincia25_Accidentes.pkl', 'RN', 'scaler_Provincia25_accidentes.sav', 'accidentes_Provincia25.weights.h5', '07/12/2020', 'Provincia25.csv'),
    26:('model_Provincia26_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia26.csv'),
    27:(0, 'No existe', 0, 0, '06/12/2020', 'Provincia27.csv'),    
    28:('model_Provincia28_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia28.csv'),
    29:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia29.csv'),
    30:('model_Provincia30_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia30.csv'),
    31:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia31.csv'),
    32:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia32.csv'),    
    33:('model_Provincia33_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia33.csv'),
    34:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia34.csv'),    
    35:('model_Provincia35_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia35.csv'),
    36:('model_Provincia36_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia36.csv'),
    37:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia37.csv'),    
    38:('model_Provincia38_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia38.csv'),
    39:('model_Provincia39_Accidentes.pkl', 'RN', 'scaler_Provincia39_accidentes.sav', 'accidentes_Provincia39.weights.h5', '07/12/2020', 'Provincia39.csv'),
    40:(0, 'No existe', 0, 0, '06/12/2020', 'Provincia40.csv'),    
    41:('model_Provincia41_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia41.csv'),
    42:(0, 'No existe', 0, 0, '06/12/2020', 'Provincia42.csv'),    
    43:('model_Provincia43_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia43.csv'),
    44:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia44.csv'),    
    45:('model_Provincia45_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia45.csv'),
    46:('model_Provincia46_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia46.csv'),
    47:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia47.csv'),
    48:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia48.csv'), 
    49:('model_Provincia49_Accidentes.pkl', 'RN', 'scaler_Provincia49_accidentes.sav', 'accidentes_Provincia49.weights.h5', '07/12/2020', 'Provincia49.csv'),
    50:('model_Provincia50_Accidentes.pkl', 'PROPHET', 0, 0, '07/12/2020', 'Provincia50.csv'),
    51:(0, 'No existe', 0, 0, '06/12/2020', 'Provincia51.csv'),
    52:(0, 'No existe', 0, 0, '07/12/2020', 'Provincia52.csv'), 
}

modelos_provincias_victimas = {
    1:('model_Provincia1_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia1.csv'),
    2:('model_Provincia2_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia2.csv'),
    3:('model_Provincia3_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia3.csv'),
    4:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia4.csv'),
    5:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia5.csv'),
    6:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia6.csv'),
    7:('model_Provincia7_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia7.csv'),
    8:('model_Provincia8_Accidentes.pkl', 'RN', 'scaler_Provincia8_accidentes.sav', 'accidentes_Provincia8.weights.h5', '07/09/2020', 'Provincia8.csv'),
    9:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia9.csv'),
    10:('model_Provincia10_Accidentes.pkl', 'RN', 'scaler_Provincia10_accidentes.sav', 'accidentes_Provincia10.weights.h5', '07/09/2020', 'Provincia10.csv'),
    11:('model_Provincia11_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia11.csv'),
    12:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia12.csv'),
    13:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia13.csv'),
    14:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia14.csv'),
    15:('model_Provincia15_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia15.csv'),
    16:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia16.csv'),
    17:('model_Provincia17_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia17.csv'),
    18:('model_Provincia18_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia18.csv'),
    19:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia19.csv'),
    20:('model_Provincia20_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia20.csv'),
    21:('model_Provincia21_Accidentes.pkl', 'RN', 'scaler_Provincia21_accidentes.sav', 'accidentes_Provincia21.weights.h5', '07/09/2020', 'Provincia21.csv'),
    22:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia22.csv'),
    23:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia23.csv'),
    24:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia24.csv'),
    25:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia25.csv'),
    26:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia26.csv'),
    27:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia27.csv'),    
    28:('model_Provincia28_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia28.csv'),
    29:('model_Provincia29_Accidentes.pkl', 'RN', 'scaler_Provincia29_accidentes.sav', 'accidentes_Provincia29.weights.h5', '07/09/2020', 'Provincia29.csv'),
    30:('model_Provincia30_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia30.csv'),
    31:('model_Provincia31_Accidentes.pkl', 'RN', 'scaler_Provincia31_accidentes.sav', 'accidentes_Provincia31.weights.h5', '07/09/2020', 'Provincia31.csv'),
    32:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia32.csv'),    
    33:('model_Provincia33_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia33.csv'),
    34:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia34.csv'),    
    35:('model_Provincia35_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia35.csv'),
    36:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia36.csv'),    
    37:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia37.csv'),    
    38:('model_Provincia38_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia38.csv'),
    39:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia39.csv'),    
    40:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia40.csv'),    
    41:('model_Provincia41_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia41.csv'),
    42:(0, 'No existe', 0, 0, '05/09/2020', 'Provincia42.csv'),  
    43:('model_Provincia43_Accidentes.pkl', 'PROPHET', 0, 0, '07/09/2020', 'Provincia43.csv'),  
    44:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia44.csv'),    
    45:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia45.csv'),    
    46:('model_Provincia46_Accidentes.pkl', 'RN', 'scaler_Provincia46_accidentes.sav', 'accidentes_Provincia46.weights.h5', '07/09/2020', 'Provincia46.csv'),
    47:('model_Provincia47_Accidentes.pkl', 'RN', 'scaler_Provincia47_accidentes.sav', 'accidentes_Provincia47.weights.h5', '07/09/2020', 'Provincia47.csv'),
    48:('model_Provincia48_Accidentes.pkl', 'RN', 'scaler_Provincia48_accidentes.sav', 'accidentes_Provincia48.weights.h5', '07/09/2020', 'Provincia48.csv'),
    49:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia49.csv'),
    50:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia50.csv'), 
    51:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia51.csv'),
    52:(0, 'No existe', 0, 0, '07/09/2020', 'Provincia52.csv'),
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
