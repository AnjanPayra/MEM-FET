# -*- coding: utf-8 -*-
"""
Created on Mon Dec 09 15:53:03 2019

@author: lab5-30
"""
# α, β, γ, δ membership based essential protein prediction

file_input2 = open('D:\\Essential protein prediction\\input\\Centrality_YDIP\\40%Cen.csv', 'r')
#ew_proteins = file_input2.readlines()
file_input3 = open('D:\\Essential protein prediction\\input\\Centrality_YDIP\\Cen_Phy_Memb.csv', 'w')
th=file_input2.readlines()
for line1 in th:
        p=q=r=0.0;
        line1=line1.strip('\n')
        store1 = line1.split(',')
        f_th = open('D:\\Essential protein prediction\\input\\Centrality_YDIP\\55phy.csv', 'r')
        for line in f_th:
            line=line.strip('\n')
            store = line.split(',')
            if(store[0]==store1[0]):
                p=store1[1]
                q=store[1]
        f_th1 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Protein_node.csv', 'r')
        for line2 in f_th1:
                line2=line2.strip('\n')
                store2 = line2.split(',')
                if(store2[0]==store1[0]):
                    r=store2[2]
               # print(str(store[1])+'||'+str(t))
        file_input3.write(store1[0]+','+str(q)+','+str(p)+','+str(r)+','+str(.25*float(q)+.25*float(r)+.5*float(p))+','+str(.2*float(q)+.2*float(r)+.6*float(p))+','+str(.15*float(q)+.15*float(r)+.7*float(p))+','+str(.1*float(q)+.1*float(r)+.8*float(p))+','+str(.5*float(q)+.5*float(r)+.9*float(p))+'\n')
file_input3.close()   
