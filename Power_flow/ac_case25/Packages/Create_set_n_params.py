def create_set_and_params(np,pd,pre_caldata_path,Bus_set):
    
    Buses = Bus_set
    
    # Save Bus set
    open(pre_caldata_path + 'Buses_Pyomo.csv', 'w').close()
    with open(pre_caldata_path + 'Buses_Pyomo.csv', 'a') as open_file:
        Headers= 'Buses'
        open_file.write(Headers + '\n')
        row_count=0
        for bus_i in Buses:
            row_data_text = str(bus_i)
            open_file.write(row_data_text + '\n')
            row_count+=1