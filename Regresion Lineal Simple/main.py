# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:45:14 2020

@author: Crhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values 



# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)


# Crear Modelo de de Regresion Lineal con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regre = LinearRegression()
regre.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regre.predict(X_test)

# Visualizar los datos de entrenamiernto
import matplotlib.pyplot as plt

plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regre.predict(X_train))
plt.title("Sueldo vs Año de experiencia [Conjunto de entrenamiento]")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,regre.predict(X_test))
plt.title("Sueldo vs Año de experiencia [Resultado de Testing]")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo ($)")









