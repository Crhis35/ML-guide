# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:21:53 2020

@author: Crhis
"""

#Apriori

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
    
# Entrenar el algoritmo de Apriori
from apriory import apriori
 # MIN_SUPPORT 3(COMPRAS MINIMAS AL DIA)* 7(DATA SET DE UNA SEMANA) /7500(TOTAL DATOS)
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Visualización de los resultados
results = list(rules)

print(results[4])