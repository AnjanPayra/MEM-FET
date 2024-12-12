# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 16:11:20 2019

@author: lab5-30
"""

#import numpy as np
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Ess_data.csv')
X = dataset.iloc[0:5087, 1:9].values
#file_input3 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\KMeans.xlsx', 'w')
plt.scatter(X[0:,5087],X[1:,9], label='True Position')
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    file_input3 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\KMeans'+i+'.csv', 'w')
    print(kmeans.labels_)
    file_input3.write(kmeans.labels_)
    file_input3.close()
    wcss.append(kmeans.inertia_)
    print(wcss)
    file_input3.write(wcss)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

'''


# K-means clustering algorithm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
sns.set()
#distortions = []
df = pd.read_csv('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Ess_data.csv')
x = df.iloc[:,[1,2,3,4,5,6,7,8,9]].values
x1 = df.iloc[:,:].values

for i in range(1, 11):
    file_input3 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\KMeans1'+str(i)+'.csv', 'w')
    kmeans5 = KMeans(n_clusters=i).fit(x)
    clusters=x1.copy()
   # clusters['pred']=kmeans5.fit_predict(x)
    file_input3.write(str(clusters)+" "+str(kmeans5.fit_predict(x)))
    print(kmeans5.fit_predict(x))
    print(clusters)
    file_input3.close()
    y_kmeans5 = kmeans5.fit_predict(x)
    print(kmeans5.labels_)
    file_input3 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\KMeans1'+str(i)+'.txt', 'w')
    print(y_kmeans5)
   # distortions.append(y_kmeans5)
    file_input3.write(str(y_kmeans5))
    file_input3.close()
    




