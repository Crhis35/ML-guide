

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./Data.csv')

# iloc[filas,columnas]
X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values

# manejar NAs

from sklearn.impute import SimpleImputer

# axis = columnas = 0, filas = 1
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X[:,1:])

X[:,1:] = imputer.transform(X[:,1:])

# Codificar datos catagoricos
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X = LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
le_y = LabelEncoder()
y = le_X.fit_transform(y)

from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Dividir training y testing del dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =  train_test_split(X,y,test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)







