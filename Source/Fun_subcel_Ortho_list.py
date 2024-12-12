# Ortho_sim_loc list

f_th = open('D:\\Essential protein prediction\\Orthologous\\YDIP1.csv', 'r')
file_input3 = open('D:\\Essential protein prediction\\Orthologous\\Ortho_FunGO_Sub.csv', 'w')
for line in f_th:
    line=line.strip('\n')
    store = line.split(',')
    file_input3.write(store[0])
    f_th1 = open('D:\\Essential protein prediction\\Orthologous\\Ortho_result.csv', 'r')
    for line1 in f_th1:
        line1=line1.strip('\n')
        store1 = line1.split(',')
        if(store[0]==store1[0]):
            file_input3.write(','+store1[3])
            break
        else:continue
    f_th2 = open('D:\\Essential protein prediction\\Orthologous\\Fun_GO.csv', 'r')
    for line2 in f_th2:
        line2=line2.strip('\n')
        store2 = line2.split(',')
        if(store[0]==store2[0]):
            file_input3.write(','+store2[1])
            break
        else:continue
    f_th3 = open('D:\\Essential protein prediction\\Orthologous\\sub_cell_update.csv', 'r')
    for line3 in f_th3:
        line3=line3.strip('\n')
        store3 = line3.split(',')
        if(store[0]==store3[0]):
            file_input3.write(','+store3[2])
            break
        else:continue
    file_input3.write('\n')        