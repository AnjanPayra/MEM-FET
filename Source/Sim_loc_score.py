# -*- coding: utf-8 -*-
"""
Created on Thu Jan 02 17:08:04 2020

@author: lab5-30
"""
#sim_loc values
file_input2 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KMeans10.csv', 'r')
#ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc1.csv', 'w')
th=file_input2.readlines()
for line1 in th:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        f_th = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\Sub_cell.csv', 'r')
        for line in f_th:
            line=line.strip('\n')
            store = line.split(',')
            if(store[0]==store1[1]):
                file_input3.write(line1+","+store[2]+","+str(float(store[2])*float(store1[12]))+","+str(float(store[2])+float(store1[12]))+"\n")
file_input3.close()                