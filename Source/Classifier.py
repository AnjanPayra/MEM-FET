# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:50:50 2021

@author: lab5-30
"""
import pandas
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder
# load data
data = pandas.read_csv('E:\\New_ECC\\ML\\YDIP_ESS_GO_SL_Entire.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,1:4]
print(X)
y = dataset[:,3]
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
data = pandas.read_csv('D:\\reema\\New_ECC\\ML\\Dataframe\\YDIP_ECC_CC_GO_SL_df.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,:4]
Y = dataset[:,3]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 5 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, label_encoded_y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)




# Recursive Feature Elimination  Centrality
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
# load data
data = pandas.read_csv('E:\\New_ECC\\ML\\YDIP_ESS_GO_SL_Entire.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,1:4]
Y = dataset[:,3]
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, label_encoded_y)
# display the relative importance of each attribute
print(model.feature_importances_)





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

file_output1=open('D:\\reema\\New_ECC\\ML\Dataset\\YMIPS_results.txt','w')
df = pd.read_csv('D:\\reema\\New_ECC\\ML\Dataset\\YMIPS_ESS_GO_SL_Entire1.csv')

y = df['Proteins']
print(y)
X = df.drop('Proteins', axis=1)
#print(X)

# Create a list of feature names
feat_labels = ['ECC','GO','Subcell']
#X[0:5]
#y

# Split the data into 20% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

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
file_output1.write('3 features:'+str(c)+'\n')

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (5 Features) Model
f=accuracy_score(y_test, y_important_pred)
file_output1.write('3 features:'+str(f)+'\n')

file_output1.close()





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

iris = pd.read_csv("D:\\reema\\New_ECC\\ML\Dataset\\YMIPS_ESS_GO_SL_Entire1.csv")
iris.head()

x = iris.iloc[:,1:4].values
y = iris['Proteins'].values

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42)

def distance(pa,pb):
    return np.sum((pa-pb)**2)**0.5

def KNN(x, y, x_query, k):
    m = x.shape[0]
    #get 100 vales
    
    distances = []
    
    #iterate over all examples
    for i in range(m):
        dis = distance(x_query, x[i]) 
        print(x[i],x_query)
        distances.append((dis, y[i]))

    
    # sort
    distances = sorted(distances)
    
    #take top 5
    distances = distances[:k]
    
    #convert to numpy to extract data
    distances = np.array(distances)
    
    labels = distances[:, 1]
    
    uniq_label, counts = np.unique(labels, return_counts=True)
    pred = uniq_label[counts.argmax()]
    
    return (pred)


prediction = []
for i in range(30):
    p = KNN(x_train, y_train, x_test[i], k =3)
    prediction.append(p)


predictions = np.array(prediction)
(y_test[:100] == predictions).sum()/len(predictions)

