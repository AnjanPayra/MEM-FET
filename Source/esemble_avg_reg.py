# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:48:40 2021

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


#Five Data Points:
# if len(data[namesClassified[0]]) != 0:
#     itemAt0 = le.fit_transform((data[namesClassified[0]]))
# else:
#     itemAt0 = emptyArray


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
# if len(data[namesClassified[5]]) != 0:
#       itemAt5 = le.fit_transform((data[namesClassified[5]]))
# else:
#     itemAt5 = emptyArray
# #Mandatory Prediction Column 7:
# if len(data[namesClassified[6]]) != 0:
#     itemAt6 = le.fit_transform((data[namesClassified[6]]))

predict = itemAt3

x = list(zip( itemAt1, itemAt2, itemAt3))
y = list(itemAt4)

#Allows user to partition the test group
testSize = input("Enter your test size: ")
# getting target data from the dataframe

 
# Splitting between train data into training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(testSize))
 
# initializing all the model objects with default parameters
model_1 = LinearRegression()
#model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()
 
# training all the model on the training dataset
model_1.fit(x_train, y_test)
#model_2.fit(X_train, y_test)
model_3.fit(x_train, y_test)
 
# predicting the output on the validation dataset
pred_1 = model_1.predict(x_test)
#pred_2 = model_2.predict(X_test)
pred_3 = model_3.predict(x_test)
 
# final prediction after averaging on the prediction of all 3 models
pred_final = (pred_1+pred_3)/2.0

print(pred_final)
 
# printing the root mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred_final))