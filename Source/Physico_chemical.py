# -*- coding: utf-8 -*-
"""
Created on Mon Aug 05 12:23:48 2019

@author: lab5-30
"""

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO

#file_input=open('C:\\Users\\Sovan\\PycharmProjects\\Funpred2\\input\\entry locus mapping.txt','r')
#prtns=file_input.readlines()

file_output1=open('D:\\Essential protein prediction\\input\\YDIP\\Physico_chemical.txt','w')
#file_output1.writelines('aromacity' + '|' + 'gravy' + '|' + 'instability index' + '|' + 'isoelectric point' + '|' + 'negatively charged particle' + '|' + 'positively charged particle' + '|' + 'extinction coefficient' + '|' + 'aliphatic index' + '|' + 'absorbance' + '|' + 'ip/mol weight' + '|' + 'proteins' + '|' + '\n')
count=0

fh = open("D:\\Essential protein prediction\\input\\YDIP\\YDIP.fasta")
for record1 in SeqIO.parse(fh, "fasta"):
    id = record1.id
    count+=1
    str1=id.split('|')
    s=''
    s+=str1[1]
    #print("s:"+s)
    file_input=open('D:\\Essential protein prediction\\input\\YDIP\\entry locus mapping.txt','r')
    prtns=file_input.readlines()
    for line in prtns:
        line=line.strip('\n')
        proteins=''
        #print("Line"+line)
        str2=line.split('	')
        s1=''
        s1+=str2[0]
        if s in s1:
            proteins=''
            proteins+=str2[1]
           # print(proteins) 
           # seq1 = record1.seq
          #  y=str(seq1)
          #  print(str(count) + '|' + id + '|' + proteins)
    #print(id)
            seq1 = record1.seq
    #y=str(seq1)
    #print(y)
            X=ProteinAnalysis(str(seq1))
    
            file_output1.writelines(proteins +'|')  
    #aromacity
            file_output1.writelines(str(round(X.aromaticity(),5)) + '|')
    #gravy
            file_output1.writelines(str(round(X.gravy(),5)) + '|')
    #instability index
            file_output1.writelines(str(round(X.instability_index(),5)) + '|')
    #isoelectric point
            file_output1.writelines(str(round(X.isoelectric_point(),5)) + '|')
    #no of -vely charged residues
            file_output1.writelines(str(round((X.count_amino_acids()['D'] + X.count_amino_acids()['E']),5)) + '|')
    #no of +vely charged residues
            file_output1.writelines(str(round((X.count_amino_acids()['R'] + X.count_amino_acids()['K']),5)) + '|')
    #extinction coefficient
            file_output1.writelines(str(round((X.count_amino_acids()['Y']*1490)+(X.count_amino_acids()['W']*5500)+(X.count_amino_acids()['C']*125),5)) + '|')
    #aliphatic index
            num_residues = len(seq1)
            xala=(X.count_amino_acids()['A']/num_residues)*100
            xval=(X.count_amino_acids()['V']/num_residues)*100
            xile=(X.count_amino_acids()['I']/num_residues)*100
            xleu=(X.count_amino_acids()['L']/num_residues)*100
            file_output1.writelines(str(round((xala + (2.9 * xval) + (3.9 * (xile + xleu))),5)) + '|')
    #absorbance
            mol_wght=X.molecular_weight()
            ext_cof=(X.count_amino_acids()['Y']*1490)+(X.count_amino_acids()['W']*5500)+(X.count_amino_acids()['C']*125)
            absrbance=(float)(ext_cof/mol_wght)
            file_output1.writelines(str(round(absrbance,5)) + '|')
    #compute ip/molecular weight
            ip_mol_wght=(float)(X.isoelectric_point()/X.molecular_weight())
            file_output1.writelines(str(round(ip_mol_wght,5)) + '\n')
    #file_output1.writelines(proteins +'\n')
            file_output1.flush()
fh.close()