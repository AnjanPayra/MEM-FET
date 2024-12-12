# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:13 2019

@author: lab5-30
"""
# Construction of essential dataset

file_input2 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Centrlity_Ess.csv', 'r')
#ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Ess_data.csv', 'w')
th=file_input2.readlines()
for line1 in th:
        #p=q=r=0.0;
        line1=line1.strip('\n')
        store1 = line1.split(',')
        f_th = open('D:\\Essential protein prediction\\input\\Priority _based_Similarity_measure\\Physico_chemical_ess.csv', 'r')
        for line in f_th:
            line=line.strip('\n')
            store = line.split(',')
            if(store[0]==store1[0]):
                     file_input3.write(line1+","+line+"\n")
file_input3.close()   
