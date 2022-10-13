# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:04:30 2022

@author: PAOLA ANDREA REALPE
"""

import numpy as np
import pandas as pd

#Importar base de datos
dataset = pd.read_csv('AccidentesBarranquilla.csv')

#Variable dependiente
Y=dataset.iloc[:, 4].values
#Variables independientes
X=dataset.iloc[:,[1, 2, 3, 5, 6, 7]].values


#Convirtiendo datos categoricos
#Importando sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#Convertir las variables de cadena a enteros
X[:,0]=LabelEncoder().fit_transform(X[:,0])
X[:,2]=LabelEncoder().fit_transform(X[:,2])
X[:,3]=LabelEncoder().fit_transform(X[:,3])
X[:,4]=LabelEncoder().fit_transform(X[:,4])
X[:,5]=LabelEncoder().fit_transform(X[:,5])


#cambio de los datos categoricos en la matriz x columna 1
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
#reemplazar los datos categoricos
X=np.array(ct.fit_transform(X), dtype=np.float)


######### REDES NEURONALES #########

#Separando modelo de entrenamiento y de pruebas
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
                                                    
#Escalando categorias
from sklearn.preprocessing import StandardScaler

#Objeto que realiza el escalado en x
sc_X = StandardScaler()
#Cambio de los datos de x test y x train
X_test=sc_X.fit_transform(X_test)
X_train=sc_X.fit_transform(X_train)


#Crear red
#theano - tensorflow - keras
import keras
from keras.models import Sequential 
from keras.layers import Dense

#Inicializar la red neuronal
classifier = Sequential()

#Agregar la capa input y la primera capa oculta
classifier.add(Dense(units=5,kernel_initializer='uniform',activation='relu',input_dim=10))

#Agregar segunda capa capa oculta
classifier.add(Dense(units=5,kernel_initializer='uniform',activation='relu'))

#Capa de salida - solo se necesita 1 porque solo tenemos una variable dependiente
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compilar la red - porcentaje basado en el error
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
