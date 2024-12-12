# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:25:15 2021

@author: lab5-30
"""
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
# importing machine learning models for prediction
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, preprocessing
from sklearn.ensemble import GradientBoostingRegressor

#Opening Message
print("Hello, you will find instructions for building your model in the README.md file.")

#User selects a .csv file from the data folder
fileIn = 'D:\\reema\\New_ECC\\ML\\Dataset\\KNN.csv'

data = pd.read_csv(fileIn)
namesInCsv = pd.read_csv(fileIn, nrows=0)
namesClassified = []

for name in namesInCsv:
    namesClassified.append(name)

#Empty numpy array to replace empty columns
emptyColumn = []
emptyArray = np.asarray(emptyColumn)

#Future Update: This section will be of variable length
le = preprocessing.LabelEncoder()


if len(data[namesClassified[1]]) != 0:
    itemAt1 = le.fit_transform((data[namesClassified[1]]))
else:
    itemAt1 = emptyArray
if len(data[namesClassified[2]]) != 0:
    itemAt2 = le.fit_transform((data[namesClassified[2]]))
else:
    itemAt2 = emptyArray
if len(data[namesClassified[3]]) != 0:
    itemAt3 = le.fit_transform((data[namesClassified[3]]))
else:
    itemAt3 = emptyArray
if len(data[namesClassified[4]]) != 0:
    itemAt4 = le.fit_transform((data[namesClassified[4]]))
else:
    itemAt4 = emptyArray

predict = itemAt3

x = list(zip( itemAt1, itemAt2, itemAt3))
y = list(itemAt4)

#Allows user to partition the test group
testSize = input("Enter your test size: ")
# getting target data from the dataframe

 
# Splitting between train data into training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(testSize))
 
model = GradientBoostingRegressor()
 
# training model
model.fit(x_train, y_train)
 
# predicting the output on the test dataset
pred = model.predict(x_test)
 
# printing the root mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred))