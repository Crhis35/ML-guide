# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 12:21:16 2020

@author: Crhis
"""

# Natural Languaje Processing

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar dataset

dataset = pd.read_csv("./NLP/Restaurant_Reviews.tsv",
                      delimiter="\t", quoting=3)

# Limpiar datos
nltk.download('stopwords')

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Crear el Bag of Words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)


# Ajustar el clasificador en el Conjunto de Entrenamiento
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
# classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test, y_pred)

print(cm)
#print("Presicion: ",74+23/(74+23+39+64))
