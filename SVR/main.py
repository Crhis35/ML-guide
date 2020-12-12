# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:23:23 2020

@author: Crhis
"""

# Support Vector Machine


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))
y = np.ravel(y)

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf") # Radial Base Function (Gaussian)
regression.fit(X,y)

# Predicción de nuestros modelos con SVR
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)

# Visualización de los resultados del Modelo SVR
plt.title("Modelo de Regresión")
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regression.predict(X)), color = "blue")
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()