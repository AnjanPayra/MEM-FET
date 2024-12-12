# -*- coding: utf-8 -*-
"""
Created on Thu Sep 05 12:09:21 2019

@author: lab5-30
"""
# Subcellular localization
#file read
file_input = open('D:\\Essential protein prediction\\input\\Sub_cellular.txt', 'r')
#file write
file_output1 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\nuclear_node.txt', 'w')
file_output2 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\cytoplasm_node.txt', 'w')
file_output3 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Peripheral_node.txt', 'w')
file_output4 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Golgi_node.txt', 'w')
file_output5 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Endo_node.txt', 'w')
file_output6 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Vacuole_node.txt', 'w')
file_output7 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Bud_node.txt', 'w')
file_output9 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\cell_wall_node.txt', 'w')
file_output10 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\peroxisome_node.txt', 'w')
file_output11 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\mitrochondrion_node.txt', 'w')
file_output12 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\cytosol_node.txt', 'w')

file_output8 = open('D:\\Essential protein prediction\\input\\output\\Sub_cel\\Protein_node.csv', 'w')

#read all lines and directly store in list hold
hold = file_input.readlines()
# print total number of lines in a file
print(len(hold))
#access each line in hold
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0
count11=0
for line in hold:
    #split line on base of |
    splitt = line.split('\t')
    #write original pairs with functions into a file i.e. three_field
    #print(splitt[0] + '|' + splitt[1] + '|' + splitt[2]+'\n')
    #file_output1.write(splitt[0] + '|' + splitt[1] + '|' + splitt[2]+'\n')
   # if len(splitt[2])==0 or len(splitt[2])==1 :
      #  file_output4.write(splitt[1]+'\n')
        
    if 'Nucle' in splitt[1] or 'nucle' in splitt[1]:
        file_output1.write(splitt[0]+'\n')
        count1=count1+1
        
        
    if 'Cytopl'in splitt[1] or 'cytopl'in splitt[1]:
        file_output2.write(splitt[0]+'\n')
        count2=count2+1
       
    if 'Peripheral'in splitt[1] or 'peripheral'in splitt[1]:
        file_output3.write(splitt[0]+'\n')
        count3=count3+1
    
    if 'Golgi'in splitt[1] or 'golgi'in splitt[1]:
        file_output4.write(splitt[0]+'\n')
        count4=count4+1
        
    if 'Endo'in splitt[1] or 'endo'in splitt[1]:
        file_output5.write(splitt[0]+'\n')
        count5=count5+1
    
    if 'Vacuole'in splitt[1] or 'vacuole'in splitt[1]:
        file_output6.write(splitt[0]+'\n')
        count6=count6+1
        
    if 'Bud'in splitt[1] or 'Bud'in splitt[1]:
        file_output7.write(splitt[0]+'\n')
        count7=count7+1
        
    if 'Cell membrane'in splitt[1] or 'cell membrane'in splitt[1]:
        file_output9.write(splitt[0]+'\n')
        count8=count8+1
        
    if 'Peroxisome'in splitt[1] or 'peroxisome'in splitt[1]:
        file_output10.write(splitt[0]+'\n')
        count9=count9+1
     
    if 'Mitro'in splitt[1] or 'mitro'in splitt[1]:
        file_output11.write(splitt[0]+'\n')
        count10=count10+1
        
    if 'Cytosol'in splitt[1] or 'cytosol'in splitt[1]:
        file_output12.write(splitt[0]+'\n')
        count11=count11+1
        
    #if 'Nucle' not in splitt[2] and 'nucle' not in splitt[2] and 'Cytopl' not in splitt[2] and 'cytopl' not in splitt[2] and len(splitt[2])!=0 and len(splitt[2])!=1:
      #  file_output3.write(splitt[1]+'\n')
        
#file close
file_output1.close()
file_output2.close()
file_output3.close()
file_output4.close()
file_output5.close()
file_output6.close()
file_output7.close()
file_output9.close()
file_output10.close()
file_output11.close()
file_output12.close()
file_input.close()

# protein wise protein membership value

file_input = open('D:\\Essential protein prediction\\input\\Sub_cellular.txt', 'r')
hold = file_input.readlines()

for line in hold:
    #split line on base of |
    s=0.0
    m=0.0
    n=0.0
    splitt = line.split('\t')
    
        
    if 'Nucle' in splitt[1] or 'nucle' in splitt[1]:
        s=s+(1.0/count1)
        m=m+(count1/4098.0)
        n=n+(1.0/count1)
        
         
    if 'Cytopl'in splitt[1] or 'cytopl'in splitt[1]:
        s=s+(1.0/count2)
        m=m+(count2/4098.0)
       
    if 'Peripheral'in splitt[1] or 'peripheral'in splitt[1]:
        s=s+(1.0/count3)
        m=m+(count3/4098.0)
    
    if 'Golgi'in splitt[1] or 'golgi'in splitt[1]:
        s=s+(1.0/count4)
        m=m+(count4/4098.0)
        
    if 'Endo'in splitt[1] or 'endo'in splitt[1]:
        s=s+(1.0/count5)
        m=m+(count5/4098.0)
        n=n+(1.0/count5)
    
    if 'Vacuole'in splitt[1] or 'vacuole'in splitt[1]:
        s=s+(1.0/count6)
        m=m+(count6/4098.0)
        
    if 'Bud'in splitt[1] or 'Bud'in splitt[1]:
        s=s+(1.0/count7)
        m=m+(count7/4098.0)
        
    if 'Cell membrane'in splitt[1] or 'cell membrane'in splitt[1]:
        s=s+(1.0/count8)
        m=m+(count8/4098.0)
        
    if 'Peroxisome'in splitt[1] or 'peroxisome'in splitt[1]:
        s=s+(1.0/count9)
        m=m+(count9/4098.0)
        
    if 'Mitro'in splitt[1] or 'mitro'in splitt[1]:
        s=s+(1.0/count10)
        m=m+(count10/4098.0)
        n=n+(1.0/count10)
        
    if 'Cytosol'in splitt[1] or 'cytosol'in splitt[1]:
        s=s+(1.0/count11)
        m=m+(count11/4098.0)
        n=n+(1.0/count11)
        
    file_output8.write(splitt[0]+','+str(s)+','+str(m)+','+str(n)+'\n')
    
file_output8.close()
file_input.close()
