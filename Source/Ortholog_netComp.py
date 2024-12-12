# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:03:28 2020

@author: lab5-30
"""
#Orthologous score in a protein complex

f_th = open('D:\\Essential protein prediction\\Orthologous\\YDIP1.csv', 'r')
file_input3 = open('D:\\Essential protein prediction\\Orthologous\\YDIP_level1.csv', 'w')
for line in f_th:
    line=line.strip('\n')
    file_input3.write(line)
    f_th1 = open('D:\\Essential protein prediction\\Orthologous\\YDIP.csv', 'r')
    for line1 in f_th1:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        if(line==store1[0]):
            file_input3.write(","+store1[1])
        if(line==store1[1]):
            file_input3.write(","+store1[0])
    file_input3.write('\n')   
    
    

f_th = open('D:\\Essential protein prediction\\Orthologous\\YDIP_level1.csv', 'r')
file_input3 = open('D:\\Essential protein prediction\\Orthologous\\YDIP_level1_COG.csv', 'w')
for line in f_th:
    line=line.strip('\n')
    store = line.split(',')
    file_input3.write(store[0])
    for p in range(len(store)):
        f_th1 = open('D:\\Essential protein prediction\\Orthologous\\Orthologous_yeast.csv', 'r')
        for line1 in f_th1:
            line1=line1.strip('\n')
            store1 = line1.split(',')
            if(store[p]==store1[1]):
                file_input3.write(','+store1[0])
                break
    file_input3.write('\n')   



f_th = open('D:\\Essential protein prediction\\Orthologous\\YDIP_level1_COG.csv', 'r')
file_input3 = open('D:\\Essential protein prediction\\Orthologous\\YDIP_level1_COG_AssoV.csv', 'w')
for line in f_th:
    line=line.strip('\n')
    store = line.split(',')
    file_input3.write(store[0])
    if(len(store)>2):
        for p in range(len(store)-2):
            f_th1 = open('D:\\Essential protein prediction\\Orthologous\\COG_11.csv', 'r')
            for line1 in f_th1:
                line1=line1.strip('\n')
                store1 = line1.split(',')
                if(store[1]==store1[0]):
                    if(store[p+2]==store1[1]):
                        file_input3.write(','+store1[2])
                        break
                elif(store[1]==store1[1]):
                    if(store[p+2]==store1[0]):
                        file_input3.write(','+store1[2])
                        break
                else:continue
    file_input3.write('\n')                          