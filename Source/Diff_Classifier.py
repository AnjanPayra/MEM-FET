# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:37:55 2018

@author: lab5-30
"""
'''
file_input1 = open('D:\\Papers_Essential_Proteins1\\Data_set\\Intersect\\Validate\\own_low20.txt', 'r')
ew_proteins = file_input1.readlines()
print(type(ew_proteins))
sa=set(ew_proteins)
file_input3 = open('D:\\Papers_Essential_Proteins1\\Data_set\\Intersect\\Validate\\ess.txt', 'r')
th=file_input3.readlines()
sb=set(th)
file_th = open('D:\\Papers_Essential_Proteins1\\Data_set\\Intersect\\Validate\\\\High_threshold.txt', 'w')
c = sa.intersection(sb)
print(len(c))
#print(list(c))
for  line in c:
    file_th.write(line)
print(len(c))
file_th.close()
'''


#XGBClassifier  Centrality
# plot feature importance manually

import pandas
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\All_Centrality_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:8]
print(X)
y = dataset[:,7]
print(y)
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)
# fit model no training data
print('entering model')
model = XGBClassifier()
model.fit(X, label_encoded_y)
print('finished model')
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
plot_importance(model)
pyplot.show()



#LogisticRegression  Centrality

import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\All_Centrality_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,7]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 5 attributes
rfe = RFE(model, 5)
rfe = rfe.fit(X, label_encoded_y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)




# Recursive Feature Elimination  Centrality
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\All_Centrality_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,7]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, label_encoded_y)
# display the relative importance of each attribute
print(model.feature_importances_)







#XGBClassifier  Physico-chemical Propertices
# plot feature importance manually

import pandas
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\input\\YDIP\\Physico_chemical_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:10]
print(X)
y = dataset[:,9]
print(y)
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)
# fit model no training data
print('entering model')
model = XGBClassifier()
model.fit(X, label_encoded_y)
print('finished model')
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
plot_importance(model)
pyplot.show()



#LogisticRegression  Physico-chemical Propertices

import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\input\\YDIP\\Physico_chemical_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:10]
Y = dataset[:,9]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 5 attributes
rfe = RFE(model, 5)
rfe = rfe.fit(X, label_encoded_y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)




# Recursive Feature Elimination  Physico-chemical Propertices
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
# load data
data = pandas.read_csv('D:\\Essential protein prediction\\input\\YDIP\\Physico_chemical_WOP.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:10]
Y = dataset[:,9]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, label_encoded_y)
# display the relative importance of each attribute
print(model.feature_importances_)



'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

file_output1=open('D:/Essential protein prediction/datasets/datasets/results.txt','w')
df = pd.read_csv('D:/Essential protein prediction/datasets/datasets/labelled physico chemical wth proteins and functions.csv')

y = df['functions']
print(y)
X = df.drop('functions', axis=1)
#print(X)

# Create a list of feature names
feat_labels = ['aromacity','gravy','instability index','isoelectric point','negatively charged particle','positively charged particle','extinction coefficient','aliphatic index','absorbance','ip/mol weight']
#X[0:5]
#y

# Split the data into 40% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    file_output1.write(str(feature)+'\n')
    
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf, threshold=0.10)

# Train the selector
sfm.fit(X_train, y_train)
SelectFromModel(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=-1, oob_score=False, random_state=42,
            verbose=0, warm_start=False),
        prefit=False, threshold=0.10)
# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    file_output1.write(feat_labels[feature_list_index]+'\n')
    
# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=-1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature (10 Features) Model
c=accuracy_score(y_test, y_pred)
file_output1.write('10 features:'+str(c)+'\n')

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (5 Features) Model
f=accuracy_score(y_test, y_important_pred)
file_output1.write('5 features:'+str(f)+'\n')

file_output1.close()

'''