def create_Y_bus(np,pd,pre_caldata_path,Bus_set,Branch_data,Transformer_data):
    
    Buses = Bus_set
    G_matrix = np.zeros([len(Buses),len(Buses)])
    B_matrix = np.zeros([len(Buses),len(Buses)])
    
    
    """
    Y행렬 만드는 부분 - 태호, Y행렬 만드는 부분 참고하여 입력하기 ('25.04.11)
    Branch_data 불러오면 됨
    Branch_data['From'] -> Bus i
    Branch_data['To'] -> Bus j
    Branch_data['R (pu)'] -> R
    Branch_data['X (pu)'] -> X
    Branch_data['B (pu)'] -> B
    """

    open(pre_caldata_path + 'Y_bus_Pyomo.csv', 'w').close()
    with open(pre_caldata_path + 'Y_bus_Pyomo.csv', 'a') as open_file:
        Headers= 'Bus_i,Bus_j,Bus_G,Bus_B'
        open_file.write(Headers + '\n')
        row_count=0
        for bus_i in Buses:
            col_count=0
            for bus_j in Buses:
                row_data_text = str(bus_i) + ',' + str(bus_j) + ',' + str(G_matrix[row_count,col_count]) + ',' + str(B_matrix[row_count,col_count])
                open_file.write(row_data_text + '\n')
                col_count+=1
            row_count+=1
    