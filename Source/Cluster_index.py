#cluster validations

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.metrics import pairwise_distances
from amltlearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score
  
# Generating the sample data from make_blobs 
  
df = pd.read_csv('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Ess_data.csv')
X = df.iloc[:,[1,2,3,4,5,6,7,8,9]].values
lscores=[]  
no_of_clusters = [2,3,4,5,6,7,8,9,10] 
  
for n_clusters in no_of_clusters: 
  
    cluster = KMeans(n_clusters = n_clusters) 
    cluster_labels = cluster.fit_predict(X) 
   # y_pred = cluster.predict(df) 
  
    # The silhouette_score gives the  
    # average value for all the samples. 
    silhouette_avg = silhouette_score(X, cluster_labels) 
   # calinski_avg=calinski_harabasz_score(X, cluster_labels)
   # davies_avg=davies_bouldin_score(X, cluster_labels)    
     
  
    print("For no of clusters =", n_clusters," The average silhouette_score is :", silhouette_avg)
    print("For no of clusters =", n_clusters," The average calinski_score is :", metrics.calinski_harabasz_score(X, cluster_labels))
    #print("For no of clusters =", n_clusters," The average davies_score is :", davies_avg)
   