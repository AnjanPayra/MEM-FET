# -*- coding: utf-8 -*-
"""
Created on Fri Jan 03 11:45:28 2020

@author: lab5-30
"""
# 3-sigma approach to find essential proteins using similarity localize score

import statistics
import math

file_input1 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc1.csv', 'r')
ew_pro = file_input1.readlines()
file_output1 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc+_Thsh.csv', 'w')
weight = []
for line in ew_pro:
    store = line.split(',')
    weight.append(float(store[16]))
arthmtc_mean = statistics.mean(weight)
stndrd_dev = statistics.stdev(weight)
print(str(arthmtc_mean)+"   "+str(stndrd_dev) )     
# k=1
low_thrshld = arthmtc_mean + (1 * stndrd_dev * (1 - (1 / (1 + math.pow(stndrd_dev, 2)))))
file_output1.write('low threshold' + ',' + str(low_thrshld) + '\n')
# k=2
medium_thrshld = arthmtc_mean + (2 * stndrd_dev * (1 - (1 / (1 + math.pow(stndrd_dev, 2)))))
file_output1.write('medium threshold' + ',' + str(medium_thrshld) + '\n')
# k=3
high_thrshld = arthmtc_mean + (3 * stndrd_dev * (1 - (1 / (1 + math.pow(stndrd_dev, 2)))))
file_output1.write('high threshold' + "," + str(high_thrshld) + '\n')
file_output1.close()




file_input2 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc1.csv', 'r')
ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc+_Thsh.csv', 'r')
th=file_input3.readlines()
for line1 in th:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        t=float(store1[1])
        print('t:'+str(t))
        f_th = open('D:\\Essential protein prediction\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Priority _based_Similarity_measure\\Val\\KM_Sim_loc+_'+store1[0]+'.csv', 'w')
        for line in ew_proteins:
            line=line.strip('\n')
            store = line.split(',')
            if(float(store[16])>t):
               # print(str(store[1])+'||'+str(t))
                f_th.write(line+'\n')
f_th.close()  