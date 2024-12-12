# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 14:51:41 2020

@author: lab5-30
"""

file_input2 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\level-1.csv', 'r')
#ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\level-2.csv', 'w')
th=file_input2.readlines()
for line1 in th:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        f_th = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\YDIP.csv', 'r')
        for line in f_th:
            line=line.strip('\n')
            store = line.split(',')
            if(store[0]==store1[0]):
                file_input3.write(store[1]+"\n")
            elif(store[1]==store1[0]):
                file_input3.write(store[0]+"\n")
file_input3.close()  



