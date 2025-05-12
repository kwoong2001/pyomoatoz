"""
250507: Matpower 용으로 변환

250501: Branch 데이터 중 Switch가 있는 성분은 _with_switch라는 함수를 사용하여 선로의 상태를 반영

250429까지: 기본적인 Bus, Branch, Gen, Load, Y Bus 생성
"""

"""
Set parameters and values
- Bus Data
- Branch Data
- Gen Data
- Load Data
- Y Bus 
- Etc 
"""

# 선로의 상태를 반영할 수 있는 변수 추가
def Set_All_Values_with_switch(np,pd,save_directory,m,mpc,previous_branch_array):
    #Bus info
    Bus_info = Set_Bus(pd,save_directory,mpc)
    #Line info
    Line_info = Set_Line_with_switch(pd,save_directory,m,mpc,previous_branch_array)
    #Gen info
    Gen_info = Set_Gen(pd,save_directory,mpc)
    #Load info
    Load_info = Set_Load(pd,save_directory,mpc)
    # Ymatrix
    Y_mat_info = Creating_Y_matrix(np,pd,save_directory,m,mpc)

    return Bus_info, Line_info, Gen_info, Load_info, Y_mat_info


def Set_Bus(pd,save_directory,mpc):
    Bus_info_index = range(1, len(mpc['bus'])+1)
    Bus_info_columns = ['Buses', 'baseKV', 'zone', 'Vmax_pu', 'Vmin_pu']
    Bus_info = pd.DataFrame(index = Bus_info_index, columns = Bus_info_columns)
    
    bus_idx = 1
    for bus in mpc['bus']:
        Bus_info.loc[bus_idx,'Buses'] = int(bus[0])
        Bus_info.loc[bus_idx,'baseKV'] = bus[9]
        Bus_info.loc[bus_idx,'zone'] = bus[10]
        Bus_info.loc[bus_idx,'Vmax_pu'] = bus[11]
        Bus_info.loc[bus_idx,'Vmin_pu'] = bus[12]
        bus_idx +=1

    tmp = Bus_info['Buses']
    tmp.to_csv(save_directory+'Buses_set_for_pyomo.csv',index=False) # For Pyomo Sets

    Bus_info.to_csv(save_directory+'Bus_info.csv',index=False)
    return Bus_info

def Set_Gen(pd,save_directory,mpc):
    mat_gen_info_columns = ['bus','Pg',	'Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q',	'apf']
    mat_gen_info = pd.DataFrame(mpc['gen'], columns = mat_gen_info_columns)
    
    #%	1	startup	shutdown	n	x1	y1	...	xn	yn
    #%	2	startup	shutdown	n	c(n-1)	...	c0
    mat_gen_cost_info_columns = ['type', 'startup',	'shutdown',	'n','c(2)','c(1)','c(0)']
    mat_gen_cost_info = pd.DataFrame(mpc['gencost'], columns = mat_gen_cost_info_columns)
    
    #Gen Data
    gen_index = range(1,mat_gen_info.shape[0]+1)
    gen_columns = ['bus','in_service','vm_pu','p_mw','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar','c(2)','c(1)','c(0)']
    gen_info = pd.DataFrame(index = gen_index, columns = gen_columns)
    
    gen_info['bus']=mat_gen_info['bus'].astype(int).values
    gen_info['in_service']=mat_gen_info['status'].astype(int).values
    gen_info['vm_pu']=mat_gen_info['Vg'].values
    gen_info['p_mw']=mat_gen_info['Pg'].values
    gen_info['max_p_mw']=mat_gen_info['Pmax'].values
    gen_info['min_p_mw']=mat_gen_info['Pmin'].values
    gen_info['min_q_mvar']=mat_gen_info['Qmin'].values
    gen_info['max_q_mvar']=mat_gen_info['Qmax'].values
    gen_info['c(2)']=mat_gen_cost_info['c(2)'].values
    gen_info['c(1)']=mat_gen_cost_info['c(1)'].values
    gen_info['c(0)']=mat_gen_cost_info['c(0)'].values
    
    
    tmp = pd.DataFrame(gen_info.index)
    tmp.columns = ['Gens'] 
    tmp.to_csv(save_directory+'Gens_set_for_pyomo.csv',index=False) # For Pyomo Sets
    
    gen_info.to_csv(save_directory+'Gen_info.csv')
    
    return gen_info

def Set_Load(pd,save_directory,mpc):
    mat_bus_info_columns = ['bus_i','type','Pd','Qd','Gs','Bs','area','Vm','Va','baseKV','zone','Vmax','Vmin']
    mat_bus_info = pd.DataFrame(mpc['bus'], columns = mat_bus_info_columns)
    
    Load_index = range(1,mat_bus_info.shape[0]+1)
    Load_column = ['bus','p_mw','q_mvar']
    Load_info = pd.DataFrame(index=Load_index, columns = Load_column)

    Load_info['bus'] = mat_bus_info['bus_i'].astype(int).values
    Load_info['p_mw'] = mat_bus_info['Pd'].values
    Load_info['q_mvar'] = mat_bus_info['Qd'].values

    Load_info.index.name = 'Load_d'
    
    tmp = pd.DataFrame(Load_info.index)
    tmp.columns = ['Loads'] 
    tmp.to_csv(save_directory+'Loads_set_for_pyomo.csv',index=False) # For Pyomo Sets

    Load_info.to_csv(save_directory+'Load_info.csv')
    
    return Load_info

def Creating_Y_matrix(np,pd,save_directory,m,mpc):
    # Make Ymatrix
    ymat = m.makeYbus(mpc)

    Y_mat_matpower = pd.DataFrame(ymat.todense())

    Y_mat_matpower.index = range(1,ymat.shape[0]+1)
    Y_mat_matpower.columns = range(1,ymat.shape[1]+1)
    Y_mat_matpower.to_csv(save_directory+'Ymat_matpower.csv')

    bus_multi_index = pd.MultiIndex.from_product(
        [range(1,ymat.shape[0]+1), range(1,ymat.shape[1]+1)],
        names=["Bus_i", "Bus_j"]
    )

    Y_mat_info = pd.DataFrame(index=bus_multi_index,columns=['Bus_G','Bus_B'])

    for i in range(1,ymat.shape[0]+1):
        for j in range(1,ymat.shape[1]+1):
            Y_mat_info.loc[(i,j),'Bus_G'] = np.real(Y_mat_matpower.loc[i,j])
            Y_mat_info.loc[(i,j),'Bus_B'] = np.imag(Y_mat_matpower.loc[i,j])

    Y_mat_info.to_csv(save_directory+'Y_mat_info.csv')
    
    return Y_mat_info


# 선로의 상태를 반영할 수 있는 변수 추가
def Set_Line_with_switch(pd,save_directory,m,mpc,previous_branch_array):
    branch_data_column = ['fbus','tbus','r_pu','x_pu','b_pu','rateA','rateB','rateC','ratio','angle','status','angmin','angmax']
    branch_data_df = pd.DataFrame(mpc['branch'],columns = branch_data_column)
    pre_branch_data_df = pd.DataFrame(previous_branch_array,columns = branch_data_column)
    
    Line_index = range(1,branch_data_df.shape[0]+1)
    Line_column = ['from_bus','to_bus','r_pu','x_pu','b_pu','in_service','in_service (initial)']
    Line_info = pd.DataFrame(index = Line_index, columns = Line_column)

    Line_info['from_bus'] = branch_data_df['fbus'].astype(int).values
    Line_info['to_bus'] = branch_data_df['tbus'].astype(int).values
    Line_info['r_pu'] = branch_data_df['r_pu'].values
    Line_info['x_pu'] = branch_data_df['x_pu'].values
    Line_info['b_pu'] = branch_data_df['b_pu'].values
    Line_info['in_service'] = branch_data_df['status'].astype(int).values
    Line_info['in_service (initial)'] = pre_branch_data_df['status'].astype(int).values
    
    Line_info.index.name = 'Line_l'
    
    tmp = pd.DataFrame(Line_info.index)
    tmp.columns = ['Lines'] 
    tmp.to_csv(save_directory+'Lines_set_for_pyomo.csv',index=False) # For Pyomo Sets

    Line_info.to_csv(save_directory+'Line_info.csv')
    
    return Line_info