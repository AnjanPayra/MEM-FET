# -*- coding: utf-8 -*-
"""
Created on Wed Jan 08 11:31:02 2020

@author: lab5-30
"""
# Essential proteins using similarity value of neigh 1-2 level.

file_input2 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\Ess_sim_loc.csv', 'r')
th=file_input2.readlines()
#ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\Ess_level_sim_loc.csv', 'w')


for line1 in th:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        s=[]
        f_th = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\YDIP.csv', 'r')
        for line in f_th:
            line=line.strip('\n')
            store = line.split(',')
            if(store1[0]==store[0]):
               f_th1 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\level1_next.csv', 'r')
               for line2 in f_th1:
                   line2=line2.strip('\n')
                   store2 = line2.split(',')
                   if(store[1]==store2[0]):
                       s.append(float(store2[1]))
                       #f_th1.close()
            elif(store1[0]==store[1]):
                 f_th1 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\Neigh-1-2\\level1_next.csv', 'r')
                 for line2 in f_th1:
                     line2=line2.strip('\n')
                     store2 = line2.split(',')
                     if(store[0]==store2[0]):
                         s.append(float(store2[1]))
                       #  f_th1.close()
        l=sum(s)               
        if(len(s)==0):
            file_input3.write(store1[0]+","+str(0.0)+"\n")
        else: 
            file_input3.write(store1[0]+","+str(max(s))+","+str(l)+","+ str(max(s)/l)+"\n")
       
file_input3.close()  