# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:19:33 2020

@author: Crhis
"""


# Regresion Polinomica
# Plantilla de Pre Procesado - Datos faltantes

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('./Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Ajustar Regresion lineal con el dataset

from sklearn.linear_model import LinearRegression

Lin_regression = LinearRegression()
Lin_regression.fit(X,y)

# Ajustar Regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regression = PolynomialFeatures(degree = 4)
X_poly = poly_regression.fit_transform(X)
Lin2_regression = LinearRegression()
Lin2_regression.fit(X_poly,y)


# Visualizacion del Modelo Lineal
plt.title("Modelo Regresion Lineal")
plt.scatter(X,y,color="red")
plt.plot(X,Lin_regression.predict(X),color="blue")
plt.xlabel("Nivel Empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizacion del Modelo Polinomico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.title("Modelo Regresion Polnomica")
plt.scatter(X,y,color="red")
# plt.plot(X,Lin2_regression.predict(X_poly))
plt.plot(X_grid,Lin2_regression.predict(poly_regression.fit_transform(X_grid)))
plt.xlabel("Nivel Empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Prediccion de Nuestros modelos
print(Lin_regression.predict([[6.5]]))
print(Lin2_regression.predict(poly_regression.fit_transform([[6.5]])))

