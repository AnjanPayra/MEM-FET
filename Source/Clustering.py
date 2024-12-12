import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
  
# Generating the sample data from make_blobs 

df = pd.read_csv('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Ess_data.csv')
X = df.iloc[:,[1,2,3,4,5,6,7,8,9]].values
#file_input3 = open('C:\\Users\\USER\\Desktop\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\KMeans\\Val_index.csv', 'w')
#file_input31=open('C:\\Users\\USER\\Desktop\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Hclust\\Val_index.csv', 'w')

#file_input3.write("No. of Cluster"+","+"silhouette"+","+"calinski"+","+"davies"+"\n")
#file_input31.write("No. of Cluster"+","+"silhouette"+","+"calinski"+","+"davies"+"\n")

no_of_clusters = [2,3,4,5,6,7,8,9,10] 
  
for n_clusters in no_of_clusters: 
  
    cluster = KMeans(n_clusters = n_clusters) 
    clusterH = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')

    cluster_labels = cluster.fit_predict(X) 
    clusterH_labels = clusterH.fit_predict(X)
    
    y_pred = cluster.predict(X) 
    yH_pred = clusterH.labels_ 
    
    pred = pd.DataFrame(y_pred)
    predH = pd.DataFrame(yH_pred)
    
   # print(pred)
    pd.set_option('display.max_rows',5088)
    pred.columns = ['Pred_Cluster'] 
    predH.columns = ['Pred_Cluster'] 
    
    prediction = pd.concat([df, pred], axis = 1) 
    predictionH = pd.concat([df, predH], axis = 1)
    
    print((prediction))
    
    file_input4 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Kmeans'+str(n_clusters)+ '.csv', 'w')
    file_input4.write(str(prediction.to_csv(line_terminator="\n")))
    file_input4.close()
    
    file_input41 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Hclust'+str(n_clusters)+ '.csv', 'w')
    file_input41.write(str(prediction.to_csv(line_terminator="\n")))
    file_input41.close()
    
    # The silhouette_score gives the  
    # average value for all the samples. 
    silhouette_avg = silhouette_score(X, cluster_labels) 
    calinski_avg=calinski_harabasz_score(X, cluster_labels)
    davies_avg=davies_bouldin_score(X, cluster_labels)  
    
    silhouette_avg1 = silhouette_score(X, clusterH.labels_  ) 
    calinski_avg1=calinski_harabasz_score(X, clusterH.labels_ )
    davies_avg1=davies_bouldin_score(X, clusterH.labels_ )
     
    file_input3.write(str(n_clusters)+","+str(silhouette_avg)+","+str(calinski_avg)+","+str(davies_avg)+"\n")
    file_input31.write(str(n_clusters)+","+str(silhouette_avg1)+","+str(calinski_avg1)+","+str(davies_avg1)+"\n")
file_input3.close()