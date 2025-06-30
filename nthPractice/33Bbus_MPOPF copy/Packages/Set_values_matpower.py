"""
250612: Set_line 수정

250611: Set_gen 수정

250602: _with_switch라는 표현 삭제

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
def Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array, T):
    #Bus info
    Bus_info = Set_Bus(pd,save_directory,mpc)
    #Line info
    Line_info = Set_Line(pd,save_directory,m,mpc,previous_branch_array)
    #Gen info
    Gen_info = Set_Gen(pd,save_directory,mpc, T)
    #Load info
    Load_info = Set_Load(pd,save_directory,mpc, T)
    # Ymatrix
    Y_mat_info = Creating_Y_matrix(np,pd,save_directory,m,mpc)
    #Time info
    Time_info = Set_Time(pd,save_directory,T)

    return Bus_info, Line_info, Gen_info, Load_info, Y_mat_info, Time_info


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

def Set_Gen(pd, save_directory, mpc, T):
    mat_gen_info_columns = ['bus','Pg','Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q','apf']
    mat_gen_info = pd.DataFrame(mpc['gen'], columns=mat_gen_info_columns)
    
    mat_gen_cost_info_columns = ['type', 'startup', 'shutdown', 'n']
    gen_columns = ['bus','in_service','vm_pu','p_mw','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar','time']
    
    if mpc['gencost'][0][0].astype(int) == 1:
        for n in range(int(mpc['gencost'][0][3])): # n
            mat_gen_cost_info_columns.append('x'+str(n+1))
            mat_gen_cost_info_columns.append('y'+str(n+1))
            gen_columns.append('x'+str(n+1))
            gen_columns.append('y'+str(n+1))
    else:
        for n in range(int(mpc['gencost'][0][3]),0,-1): # n
            mat_gen_cost_info_columns.append('c'+str(n-1))
            gen_columns.append('c'+str(n-1))
    
    mat_gen_cost_info = pd.DataFrame(mpc['gencost'], columns=mat_gen_cost_info_columns)
    
    gen_index = range(1, mat_gen_info.shape[0]+1)
    multi_index = pd.MultiIndex.from_product([gen_index, range(1, T+1)], names=['Gen_n', 'Time'])
    gen_info = pd.DataFrame(index=multi_index, columns=gen_columns)
    
    for t in range(1, T+1):
        for n in range(1, mat_gen_info.shape[0]+1):
            gen_info.loc[(n, t), 'bus'] = int(mat_gen_info.loc[n-1, 'bus'])
            gen_info.loc[(n, t), 'in_service'] = int(mat_gen_info.loc[n-1, 'status'])
            gen_info.loc[(n, t), 'vm_pu'] = mat_gen_info.loc[n-1, 'Vg']
            gen_info.loc[(n, t), 'p_mw'] = mat_gen_info.loc[n-1, 'Pg']
            gen_info.loc[(n, t), 'max_p_mw'] = mat_gen_info.loc[n-1, 'Pmax']
            gen_info.loc[(n, t), 'min_p_mw'] = mat_gen_info.loc[n-1, 'Pmin']
            gen_info.loc[(n, t), 'min_q_mvar'] = mat_gen_info.loc[n-1, 'Qmin']
            gen_info.loc[(n, t), 'max_q_mvar'] = mat_gen_info.loc[n-1, 'Qmax']
            # cost 컬럼이 실제로 존재할 때만 할당
            for cost_idx in gen_columns[gen_columns.index('max_q_mvar')+1:-1]:
                if cost_idx in mat_gen_cost_info.columns:
                    gen_info.loc[(n, t), cost_idx] = mat_gen_cost_info.loc[n-1, cost_idx]

    tmp = pd.DataFrame(gen_info.index.tolist(), columns=['Gens', 'Time'])
    tmp.to_csv(save_directory+'Gens_set_for_pyomo.csv', index=False)
    gen_info.to_csv(save_directory+'Gen_info.csv')
    return gen_info

def Set_Load(pd, save_directory, mpc, T):
    mat_bus_info_columns = ['bus_i','type','Pd','Qd','Gs','Bs','area','Vm','Va','baseKV','zone','Vmax','Vmin']
    mat_bus_info = pd.DataFrame(mpc['bus'], columns=mat_bus_info_columns)
    
    Load_index = range(1, mat_bus_info.shape[0]+1)
    Load_column = ['bus','p_mw','q_mvar']  # 'time' 컬럼 제거
    multi_index = pd.MultiIndex.from_product([Load_index, range(1, T+1)], names=['Load_d', 'Time'])
    Load_info = pd.DataFrame(index=multi_index, columns=Load_column)

    for t in range(1, T+1):
        for n in range(1, mat_bus_info.shape[0]+1):
            Load_info.loc[(n, t), 'bus'] = int(mat_bus_info.loc[n-1, 'bus_i'])
            Load_info.loc[(n, t), 'p_mw'] = mat_bus_info.loc[n-1, 'Pd']
            Load_info.loc[(n, t), 'q_mvar'] = mat_bus_info.loc[n-1, 'Qd']
            # Load_info.loc[(n, t), 'time'] = t  # 이 줄 삭제

    tmp = pd.DataFrame(Load_info.index.tolist(), columns=['Loads', 'Time'])
    tmp.to_csv(save_directory+'Loads_set_for_pyomo.csv', index=False)
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
def Set_Line(pd,save_directory,m,mpc,previous_branch_array):
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
    Line_info['rate_MVA'] = branch_data_df['rateA'].values # rateA, MVA rating A (long term rating), rateB, MVA rating B (short term rating), rateC, MVA rating C (emergency rating)
    Line_info['in_service'] = branch_data_df['status'].astype(int).values
    Line_info['in_service (initial)'] = pre_branch_data_df['status'].astype(int).values
    
    Line_info.index.name = 'Line_l'
    
    tmp = pd.DataFrame(Line_info.index)
    tmp.columns = ['Lines'] 
    tmp.to_csv(save_directory+'Lines_set_for_pyomo.csv',index=False) # For Pyomo Sets

    Line_info.to_csv(save_directory+'Line_info.csv')
    
    return Line_info

# Time 에 대한 정보 추가
def Set_Time(pd,save_directory,T):
    Time_info = pd.DataFrame({'Time': range(1, T+1)})
    Time_info.to_csv(save_directory+'Time_set_for_pyomo.csv', index=False)
    
    return Time_info
    
    