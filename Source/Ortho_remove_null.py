# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:42:35 2020

@author: lab5-30
"""

f_th = open('D:\\Essential protein prediction\\Orthologus_PPIN.csv', 'r')
file_input3 = open('D:\\Essential protein prediction\\Ortho.csv', 'w')
for line in f_th:
    line=line.strip('\n')
    store = line.split(',')
    if(len(store[2])==0):
        file_input3.write(line+"\n")
    else:
        store2=store[2].split('(')
        if(len(store2)>1):
            file_input3.write(store[0]+","+store2[1]+"\n")
        else:
            file_input3.write(line+"\n")