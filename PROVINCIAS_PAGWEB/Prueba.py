import json
import requests
import pandas as pd
import pickle
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}
df = pd.DataFrame({'fecha_viaje':['01/01/2020']})
data = df.to_json(orient='records')
resp = requests.post("http://localhost:8000/Predicciones", \
                    data = json.dumps(data),\
                    headers= header)
print('Respuesta de Servidor')
print(resp.json())