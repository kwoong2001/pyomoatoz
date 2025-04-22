"""
Set parameters and values
- Bus Data
- Branch Data
- Gen Data
- Load Data
- Y Bus 
- Etc 
"""
def Set_All_Values(np,pd,save_directory,net):
    #Bus info
    Bus_info = Set_Bus(pd,save_directory,net)
    #Line info
    Line_info = Set_Line(pd,save_directory,net)
    #Gen info
    Gen_info = Set_Gen(pd,save_directory,net)
    #Load info
    Load_info = Set_Load(pd,save_directory,net)
    # Ymatrix
    Y_mat_info = Creating_Y_matrix(np,pd,save_directory,net)

    return Bus_info, Line_info, Gen_info, Load_info, Y_mat_info

def Set_Bus(pd,save_directory,net):
    Bus_info = pd.DataFrame(net.bus[['name','vn_kv','max_vm_pu','min_vm_pu','type','zone','geo']])
    if 0 == Bus_info['name'][0]:
        Bus_info['name'] = Bus_info['name'].values + 1

    tmp = Bus_info['name']
    tmp.name = 'Buses' 
    tmp.to_csv(save_directory+'Buses_set_for_pyomo.csv',index=False) # For Pyomo Sets
    
    Bus_info.set_index('name',inplace=True)
    Bus_info.index.name = 'Bus_i'
    Bus_info.to_csv(save_directory+'Bus_info.csv')
    return Bus_info

def Set_Line(pd,save_directory,net):
    Line_column = ['from_bus','to_bus','r_ohm','x_ohm','c_nf','in_service','max_i_ka','max_loading_percent']
    Line_info = pd.DataFrame(columns = Line_column)

    Line_info['from_bus'] = net.line['from_bus'].values +1
    Line_info['to_bus'] = net.line['to_bus'].values +1
    Line_info['r_ohm'] = net.line['length_km'].values * net.line['r_ohm_per_km'].values
    Line_info['x_ohm'] = net.line['length_km'].values * net.line['x_ohm_per_km'].values
    Line_info['c_nf'] = net.line['length_km'].values * net.line['c_nf_per_km'].values
    Line_info['in_service'] = net.line['in_service']
    Line_info['max_i_ka'] = net.line['max_i_ka']
    Line_info['max_loading_percent'] = net.line['max_loading_percent']

    Line_info.index.name = 'Line_l'
    Line_info.index = Line_info.index + 1
    Line_info

    Line_info.to_csv(save_directory+'Line_info.csv')
    
    return Line_info

def Set_Gen(pd,save_directory,net):
    #Gen Data
    gen_columns = ['bus','in_service','vm_pu','p_mw','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar']
    gen_info = pd.DataFrame(columns = gen_columns)
    try:
        # 기본 발전 데이터
        gen_info = net.gen[['bus','in_service','vm_pu','p_mw','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar']]

        # 발전 비용함수 추가
        gen_info['cp0_eur']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cp0_eur']
        gen_info['cp1_eur_per_mw']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cp1_eur_per_mw']
        gen_info['cp2_eur_per_mw2']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cp2_eur_per_mw2']

        gen_info['cq0_eur']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cq0_eur']
        gen_info['cq1_eur_per_mvar']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cq1_eur_per_mvar']
        gen_info['cq2_eur_per_mvar2']=net.poly_cost[net.poly_cost['et'] == 'gen'].reset_index(drop=True)['cq2_eur_per_mvar2']

        tmp = gen_info['bus'].values + 1
        gen_info['bus'] = tmp
        
    except:
        print("Check genator info")

    # Slack 모선 데이터 - Slack 모선이 발전기인 경우
    slack_info = pd.DataFrame(net.ext_grid[['bus','in_service','vm_pu','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar']])
    slack_info['p_mw'] = 0
    slack_info = slack_info[['bus','in_service','vm_pu','p_mw','max_p_mw','min_p_mw','min_q_mvar','max_q_mvar']]

    # 발전 비용함수 추가
    slack_info['cp0_eur']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cp0_eur']
    slack_info['cp1_eur_per_mw']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cp1_eur_per_mw']
    slack_info['cp2_eur_per_mw2']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cp2_eur_per_mw2']

    slack_info['cq0_eur']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cq0_eur']
    slack_info['cq1_eur_per_mvar']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cq1_eur_per_mvar']
    slack_info['cq2_eur_per_mvar2']=net.poly_cost[net.poly_cost['et'] == 'ext_grid'].reset_index(drop=True)['cq2_eur_per_mvar2']

    tmp = slack_info['bus'].values + 1
    slack_info['bus']=tmp

    try:
        gen_info = pd.concat([gen_info,slack_info])
        gen_info.sort_values(by=['bus'],axis=0,inplace=True)
        gen_info.reset_index(inplace=True,drop=True)
        gen_info.index = gen_info.index + 1
        gen_info.index.name = 'G_n'
    except:
        gen_info = slack_info.copy()
        gen_info.reset_index(inplace=True,drop=True)
        gen_info.index = gen_info.index + 1
        gen_info.index.name = 'G_n'

    gen_info.to_csv(save_directory+'Gen_info.csv')
    
    return gen_info

def Set_Load(pd,save_directory,net):
    Load_column = ['bus','p_mw','q_mvar','in_service']
    Load_info = pd.DataFrame(columns = Load_column)
    if 0 == net.load['bus'][0]:
        Load_info['bus']=net.load['bus'] + 1
    else:
        Load_info['bus']=net.load['bus']
    Load_info['p_mw'] = net.load['p_mw']
    Load_info['q_mvar'] = net.load['q_mvar']
    Load_info['in_service'] = net.load['in_service']

    Load_info.index.name = 'Load_d'
    Load_info.index=Load_info.index+1

    Load_info.to_csv(save_directory+'Load_info.csv')
    
    return Load_info

def Creating_Y_matrix(np,pd,save_directory,net):
    ymat = net._ppc['internal']['Ybus'].todense()
    Y_mat_panda = pd.DataFrame(ymat)
    if 0 == net.bus['name'][0]:
        bus_index = net.bus['name'].values + 1
    else:
        bus_index = net.bus['name'].values 

    Y_mat_panda.index = bus_index
    Y_mat_panda.columns = bus_index
    Y_mat_panda.to_csv(save_directory+'Ymat_panda.csv')

    bus_multi_index = pd.MultiIndex.from_product(
        [bus_index, bus_index],
        names=["Bus_i", "Bus_j"]
    )

    Y_mat_info = pd.DataFrame(index=bus_multi_index,columns=['Bus_G','Bus_B'])

    for i in bus_index:
        for j in bus_index:
            Y_mat_info.loc[(i,j),'Bus_G'] = np.real(Y_mat_panda.loc[i,j])
            Y_mat_info.loc[(i,j),'Bus_B'] = np.imag(Y_mat_panda.loc[i,j])

    Y_mat_info.to_csv(save_directory+'Y_mat_info.csv')
    
    return Y_mat_info