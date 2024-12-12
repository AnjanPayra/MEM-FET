# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:57:30 2021

@author: lab5-30
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
# importing machine learning models for prediction
#import stacking
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


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

train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10
 
# performing train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
 
# performing test validation split
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
 
# initializing all the base model objects with default parameters
model_1 = LinearRegression()
model_3 = RandomForestRegressor()
 
# training all the model on the train dataset
 
# training first model
model_1.fit(x_train, y_train)
val_pred_1 = model_1.predict(x_val)
test_pred_1 = model_1.predict(x_test)
 
# converting to dataframe
val_pred_1 = pd.DataFrame(val_pred_1)
test_pred_1 = pd.DataFrame(test_pred_1)
 
 
# training third model
model_3.fit(x_train, y_train)
val_pred_3 = model_1.predict(x_val)
test_pred_3 = model_1.predict(x_test)
 
# converting to dataframe
val_pred_3 = pd.DataFrame(val_pred_3)
test_pred_3 = pd.DataFrame(test_pred_3)
 
# concatenating validation dataset along with all the predicted validation data (meta features)
df_val = pd.concat([pd.DataFrame(x_val), val_pred_1, val_pred_3], axis=1)
df_test = pd.concat([pd.DataFrame(x_test), test_pred_1,  test_pred_3], axis=1)
 
# making the final model using the meta features
final_model = LinearRegression()
final_model.fit(df_val, y_val)
 
# getting the final output
final_pred = final_model.predict(df_test)
 
#printing the root mean squared error between real value and predicted value
print(mean_squared_error(y_test, final_pred))
