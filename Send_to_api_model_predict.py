import json
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow.python.keras.models
import fastapi
import uvicorn
from sklearn.preprocessing import StandardScaler #Para normalizar datos, para quitar outliers
from sklearn.model_selection import train_test_split #Para el entrenamiento de la red
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



#print(model_name)

scaler = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
model_name = joblib.load("modelo_cinematica_inversa.keras")
model = tensorflow.keras.models.load_model(model_name)
#datos = np.array([[-0.284739,  -0.226987, 0.242325,  0.018225, 3.114223,  -0.11744]])
#print('Datos: '+ datos)
'''x_scaled = scaler.transform(datos)
#print('x_scaled'+ x_scaled)
y_pred_scaled = model.predict(x_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
#print(y_pred)'''




#entradas = process_numbers()
#print(entradas)


##----------------------------------------------------------------------------------##
app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_numbers():
    data = request.json
    numbers = data.get('numbers', [])

    if not isinstance(numbers, list) or len(numbers) != 6:
        return jsonify({'error': 'Exactly 6 numbers are required'}), 400

    try:
        x_scaled = scaler.transform(numbers)
        y_pred_scaled = model.predict(x_scaled)
        print(y_pred_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        return jsonify({'result': y_pred})
    except ValueError:
        return jsonify({'error': 'Invalid number format'}), 400

#result = json.dumps(process_numbers)

if __name__ == '__main__':
    app.run(debug=True)
