# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:21:13 2019

@author: lab5-30
"""
# Test for membership calculation
import pandas as pd
sheet = pd.read_excel("D:\\Essential protein prediction\\input\\Centrality_YDIP\\test.xlsx")
x=max(sheet['x'])
y=max(sheet['y'])
df = pd.DataFrame({'Network': sheet['x']/x,'closeness':sheet['y']/y})
print(df)
writer = pd.ExcelWriter('D:\\Essential protein prediction\\input\\Centrality_YDIP\\test1.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# Membership calculation for Physicochemical properties as selected by classifier 
import pandas as pd
sheet = pd.read_excel("D:\\Essential protein prediction\\input\\Centrality_YDIP\\test.xlsx")
x=max(sheet['negatively charged particle'])
y=max(sheet['isoelectric point'])
z=max(sheet['positively charged particle'])
m=max(sheet['absorbance'])
df = pd.DataFrame({'Proteins':sheet['Proteins'],'negatively charged particle': sheet['negatively charged particle']/x,'isoelectric point':sheet['isoelectric point']/y,'positively charged particle':sheet['positively charged particle']/z,'absorbance':sheet['absorbance']/m})
print(df)
writer = pd.ExcelWriter('D:\\Essential protein prediction\\input\\Centrality_YDIP\\test2.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# Membership calculation for Centrality properties as selected by classifier 
import pandas as pd
sheet = pd.read_excel("D:\\Essential protein prediction\\input\\Centrality_YDIP\\AllCentrality.xlsx")
x=max(sheet['Network'])
y=max(sheet['Subgraph'])
z=max(sheet['Degree'])
m=max(sheet['Information'])
n=max(sheet['Betweenness'])
df = pd.DataFrame({'Protein':sheet['Protein'],'Network': sheet['Network']/x,'Subgraph':sheet['Subgraph']/y,'Degree':sheet['Degree']/z,'Information':sheet['Information']/m,'Betweenness':sheet['Betweenness']/n})
print(df)
writer = pd.ExcelWriter('D:\\Essential protein prediction\\input\\Centrality_YDIP\\test1.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()