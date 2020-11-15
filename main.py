#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5 21:34:24 2020

@author: alfredocu
"""

# Importamos nuestras librerias.
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Utilizamos un dataset con la temperatura mínima registrada dia a dia desde hace 10 años.
data = pd.read_csv("daily-min-temperatures.csv")

# Se imprimen las primeras 5 filas del conjunto de datos.
print(data.head())

# Creamos una gráfica lineal donde muestra la temperatura con bastante ruido.
x = np.asanyarray(data["Temp"])
# plt.plot(x)

# Días de comparación.
p = 1

# Gráfica de dispersión del dia acutal contra el dia pasado.
# plt.scatter(x[p:], x[:-p])

# Podemos ver que tan correlacionados estan los datos.
print(np.corrcoef(x[p:].transpose(), x[:-p].transpose()))

# Gráfica Autocorrelasión.
pd.plotting.autocorrelation_plot(data.Temp) # .figure.savefig("Pred.eps", format="eps")

data2 = pd.DataFrame(data.Temp)

p = 1
for i in range(1, p + 1):
    data2 = pd.concat([data2, data.Temp.shift(-i)], axis = 1)
data2 = data2[:-p]

print(data2.head())

x = np.asanyarray(data2.iloc[:,1:])
y = np.asanyarray(data2.iloc[:,0])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain, ytrain)

print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

# plt.savefig("Temp.eps", format="eps")
# plt.savefig("Day.eps", format="eps")