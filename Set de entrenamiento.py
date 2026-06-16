#Definir dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler #Para normalizar datos, para quitar outliers
from sklearn.model_selection import train_test_split #Para el entrenamiento de la red
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#1 Cargar el data-set
#El archivo csv tiene tres columnas "Step TCP Position Joint Position"
dataset = pd.read_csv("C:/Users/Carlos/Downloads/DataRobot6.csv", sep=";") #indicamos que cada metadado del archivo está separado por ;
#dataset = pd.read_csv("DataRobot6.csv", sep=";")
#Dividimos las columnas originales, el input es la posicion de TCP y el output son los ángulos de los joints
TcpInput = dataset["TCP Position"].str.split(",", expand=True).astype(float)
TcpInput.columns = ['x','y','z','rx','ry','rz'] #Creamos las columnas para el cada coordenada y rotacion del TCP

#Ahora separamos en columnas los ángulos de los joints (output)
JointsOutput = dataset["Joint Position"].str.split(",", expand=True).astype(float)
JointsOutput.columns=['theta1','theta2','theta3','theta4','theta5','theta6'] #Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3

dataset_final= pd.concat([TcpInput, JointsOutput], axis=1)
#print(dataset_final.head())

#2 Hacer nuestros set de entrenamiento (80% de nuestros datos) y set de prueba (20% de nuestros datos)
#Definimos los valores x de entrada (coordenadas del TCP):
X = dataset_final[['x','y','z','rx','ry','rz']].values
#Definimos los valores y de salida (joints):
y = dataset_final[['theta1','theta2','theta3','theta4','theta5','theta6']].values
#Armamos los sets para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # randmon_state ensures to select the same proportion of random data from all the labels

#3 Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
#4 Definir el modelo de RNN
    #Definir funcion de activacion
#5 Compilar el modelo
#6 Entrenar el modelo
#7 Evaluar el modelo
#7.5 Guardar el modelo entrenado para cargarlo después
#8 Predecir
#9 Graficar resultados

# 4 Definir modelo y capas
model = Sequential()
#model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=6, activation='linear'))
#5 compilar
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#6.- Entrenamiento
model.summary()
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.1, verbose=2)

#7.- Evaluacion
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test mae: {test_mae:4f}")
print(f"Test loss: {test_mae:4f}")


# 8 Predecir con un solo input (reshaped)

print(np.expand_dims(X[1], axis=0))
X_test = np.expand_dims(X[1], axis=0)
y_pred = model.predict(X_test)
print("Predicción:", y_pred)






