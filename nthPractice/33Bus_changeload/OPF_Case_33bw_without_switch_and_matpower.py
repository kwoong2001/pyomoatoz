"""
OPF_Case_33bw_without_switch_and_matpower
- Matpower 에 기반한 OPF 구현
- 선로 switching이 없는 버젼

"""

import pandas as pd
import numpy as np
import os, sys
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from Packages.Set_values_matpower import *
from Packages.OPF_Creator_matpower import *
from pyomo import environ as pym
from matpower import start_instance
from oct2py import octave

"""
Set model and parameters with Matpower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')

simul_case = '33bus_NLP_Opt_problem_'

# Base MVA, Bus, Branch
base_MVA = mpc['baseMVA']
buses = mpc['bus']
branches = mpc['branch']
    
# Find Slackbus
Slackbus = 0

for bus_info in buses:
    if bus_info[1] == 3: 
        Slackbus = int(bus_info[0])
        
print(f"{len(buses)}-buses case, Slack bus: [{Slackbus}]")

# Change disconnected line to connected line
previous_branch_array = branches.copy()# Save disconnected lines data
    
# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info]=Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array)

"""
Create OPF model and Run Pyomo
"""
P_total_list = []
D_total_list = []
V_mag_list = []
V_ang_list = []
plfs_list = []


load_coefficient = [0.8, 0.9, 1.0, 1.1, 1.2]
for i in load_coefficient:
    current_Load_info = Load_info.copy()
    current_Load_info[['p_mw']] = current_Load_info[['p_mw']] * i

    model = OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,current_Load_info,Gen_info)

    # Create instance for OPF Model
    os.chdir(save_directory)
    instance = model.create_instance(save_directory + 'Model_data.dat')
    os.chdir(os.path.dirname(__file__))

    print('Initializing OPF model...')

    """
    NEOS 기반 Solver 활용 - 가입 필요 (무료)
    https://neos-server.org/neos/
    - 제한은 있지만 사용하는 것을 추천, IPOPT나 GLPK는 무료이지만 안정적인 활용에 어려움이 많음

    # formulate optimization model with NEOS
    os.environ['NEOS_EMAIL'] = ''
    optimizer = pyo.SolverManagerFactory('neos')
    Problem = optimizer.solve(instance, opt='knitro')

    """

    optimizer = pyo.SolverFactory('ipopt')
    optimizer.options['max_iter'] = 30000
    instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    Problem = optimizer.solve(instance, tee=True)

    #optimizer = pyo.SolverFactory('knitroampl',executable='C:/Program Files/Artelys/Knitro 14.2.0/knitroampl/knitroampl.exe') # Knitro solver 이용 시
    #instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    #Problem = optimizer.solve(instance,tee=True)

    '''
    in ubuntu 
    https://www.artelys.com/app/docs/knitro/1_introduction/installation/unix.html

    설치파일 압축해제 후 /usr/local/knitro-<ver> 에 위치
        > gunzip knitro-|release|-platformname.tar.gz
        > tar -xvf knitro-|release|-platformname.tar

    cat INSTALL 로 확인 후 license.tex 파일 두기 

    사용자만 사용가능
    ~/.bashrc에 아래 내용 추가
    > export PATH= <file_absolute_path>:$PATH
    > export LD_LIBRARY_PATH= <file_absolute_library_path>:$LD_LIBRARY_PATH
    '''
    #optimizer = pyo.SolverFactory('knitroampl',executable='/usr/local/knitro-14.2.0/knitroampl/knitroampl')
    #instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    #Problem = optimizer.solve(instance,tee=True)

    print('Solving OPF model...')


    """
    Result
    """

    print('----------------------------------------------------------------')
    print(f'Objective value = {instance.obj(): .4f}')
    P_total = 0
    D_total = 0
    for bus in Bus_info.index:
        for gen in Gen_info.index:
            if instance.PGen[gen,bus].value >= 1e-4:
                pgen = instance.PGen[gen,bus].value * base_MVA
            else:
                pgen = 0
            P_total = P_total + pgen
        
        if instance.PDem[bus].expr()>=1e-4:
            pdem = instance.PDem[bus].expr() * base_MVA
        else:
            pdem = 0
        D_total = D_total + pdem

    print('----------------------------------------------------------------')
    print('OPF Model total gen MW:', P_total)
    print('OPF Model total load MW:', D_total)

    P_total_list.append(P_total)
    D_total_list.append(D_total)
    V_mag_list.append([instance.V_mag[bus].value for bus in instance.V_mag])
    V_ang_list.append([instance.V_ang[bus].value * 180 / 3.141592 for bus in instance.V_ang])
    plfs_list.append([instance.P_line_flow_sending[line].expr() * base_MVA for line in instance.P_line_flow_sending])

# 반복문 끝난 뒤 DataFrame 변환
P_total_matrix = pd.DataFrame(P_total_list, index=[f"Load * {c}" for c in load_coefficient], columns=["P_total"])
D_total_matrix = pd.DataFrame(D_total_list, index=[f"Load * {c}" for c in load_coefficient], columns=["D_total"])
bus_numbers = list(range(1, len(V_mag_list[0]) + 1))
line_numbers = list(range(1, len(plfs_list[0]) + 1))
from_numbers = list(Line_info['from_bus'])
to_numbers = list(Line_info['to_bus'])
V_mag_matrix = pd.DataFrame(
    V_mag_list,
    index=[f"Load * {c}" for c in load_coefficient],
    columns=[f"{i}" for i in bus_numbers]
)
V_ang_matrix = pd.DataFrame(
    V_ang_list,
    index=[f"Load * {c}" for c in load_coefficient],
    columns=[f"{i}" for i in bus_numbers]
)
plfs_matrix = pd.DataFrame(
    plfs_list,
    index=[f"Load * {c}" for c in load_coefficient],
    columns=[f"L{line_numbers[i]}: {from_numbers[i]}-{to_numbers[i]}" for i in range(len(line_numbers))]
)


def format_result(df):
    return df.map(lambda x: 0 if abs(x) < 1e-3 else round(x, 3))

formatted_V_mag = format_result(V_mag_matrix)
formatted_V_ang = format_result(V_ang_matrix)

print("="*30)
print("▶ P_total_matrix")
print(format_result(P_total_matrix))

print("\n" + "="*30)
print("▶ D_total_matrix")
print(format_result(D_total_matrix))

V_complex_matrix = formatted_V_mag.astype(str) + " ∠ " + formatted_V_ang.astype(str)

print("="*30)
print("▶ V_matrix (Magnitude ∠ Angle)")
print(V_complex_matrix.T)
V_complex_matrix.T.to_csv("V_matrix_transposed.csv")

print("="*30)
print("\n" + "="*30)
print("▶ P_line_flow_sending(Transposed)")
print(format_result(plfs_matrix.T))
plfs_matrix.T.to_csv("P_line_flow_sending_transposed.csv")

# 3. 그래프 그리기 (데이터프레임 전치 후)
fig1, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

# --- 첫 번째 그래프: 버스별 전압 크기 ---
# V_mag_matrix를 전치(Transpose)하여 x축이 버스, 선이 부하계수가 되도록 함
V_mag_matrix_T = V_mag_matrix.T
V_mag_matrix_T.plot(ax=axes[0], marker='o', linestyle='-')

axes[0].set_title('V_mag (Load_coefficient)')
axes[0].set_xlabel('Bus (Bus Number)')
axes[0].set_ylabel('V_mag (p.u.)')
axes[0].grid(True)
axes[0].legend(title='Load Coefficient')
# X축 눈금을 버스 이름으로 설정 (이미 자동으로 되지만 명시적으로 설정 가능)
axes[0].set_xticks(range(len(V_mag_matrix_T.index)))
axes[0].set_xticklabels(V_mag_matrix_T.index)


# --- 두 번째 그래프: 선로별 조류 ---
# plfs_matrix를 전치(Transpose)하여 x축이 선로, 선이 부하계수가 되도록 함
plfs_matrix_T = plfs_matrix.T
plfs_matrix_T.plot(ax=axes[1], marker='s', linestyle='--')

axes[1].set_title('Power flow (Load_coefficient)')
axes[1].set_xlabel('Line (Line Number)')
axes[1].set_ylabel('P_sending (MW)')
axes[1].grid(True)
axes[1].legend(title='Load Coefficient')
# X축 눈금을 선로 이름으로 설정
axes[1].set_xticks(range(len(plfs_matrix_T.index)))
axes[1].set_xticklabels(range(1, len(plfs_list[0]) + 1))

# 그래프 레이아웃 자동 조정 및 표시
plt.tight_layout()
plt.show()

    # print('----------------------------------------------------------------')
    # branch = pd.DataFrame(previous_branch_array)

    # threshold = 1e-4

    # BUSdata = []
    # for bus in instance.V_mag:
    #     # V_mag, V_ang
    #     vmag = instance.V_mag[bus].value
    #     vang = instance.V_ang[bus].value * 180 / 3.141592  # degree
    #     # PGen, QGen 합산
    #     pg_sum = sum(
    #         instance.PGen[idx].value * base_MVA
    #         for idx in instance.PGen if idx[1] == bus and instance.PGen[idx].value is not None
    #     )
    #     qg_sum = sum(
    #         instance.QGen[idx].value * base_MVA
    #         for idx in instance.QGen if idx[1] == bus and instance.QGen[idx].value is not None
    #     )
    #     # PD, QD, P_loss, Q_loss
    #     pdem = instance.PDem[bus].expr() * base_MVA
    #     qdem = instance.QDem[bus].expr() * base_MVA
    #     def fmt(x):
    #         # None 처리, 0.0001 미만 절댓값은 '-'로
    #         if x is None or abs(x) < threshold:
    #             return '-'
    #         return f"{x:.3f}"
    #     row = {
    #         'bus':                bus,
    #         'V_mag(pu)':          fmt(vmag),
    #         'V_ang(deg)':         fmt(vang),
    #         'PG(MW)':             fmt(pg_sum),
    #         'QG(MVar)':           fmt(qg_sum),
    #         'PD(MW)':             fmt(pdem),
    #         'QD(MVar)':           fmt(qdem)
    #     }
    #     BUSdata.append(row)

    # Bd = pd.DataFrame(BUSdata)
    # print(Bd.to_string(index=False))
    # print('----------------------------------------------------------------')

    # Branchdata = []
    # for line in instance.P_line_flow_sending:
    #     plfs = instance.P_line_flow_sending[line].expr() * base_MVA
    #     qlfs = instance.Q_line_flow_sending[line].expr() * base_MVA
    #     plfr = instance.P_line_flow_receiving[line].expr() * base_MVA
    #     qlfr = instance.Q_line_flow_receiving[line].expr() * base_MVA
    #     pl = instance.P_line_loss[line].expr()
    #     ql = instance.Q_line_loss[line].expr()
        
    #     def fmt(x):
    #         if x is None or abs(x) < threshold:
    #             return '-'
    #         return f"{x:.3f}"
        
    #     row = {
    #         'line': line,
    #         'f_bus': branch.loc[line-1,0].astype(int),
    #         't_bus': branch.loc[line-1,1].astype(int),
    #         'P_s(MW)': fmt(plfs),
    #         'Q_s(MVar)': fmt(qlfs),
    #         'P_r(MW)': fmt(plfr),
    #         'Q_r(MVar)': fmt(qlfr),
    #         'P_l(MW)': fmt(pl),
    #         'Q_l(MVar)': fmt(ql)                
    #     }
    #     Branchdata.append(row)

    # Brd = pd.DataFrame(Branchdata)
    # print(Brd.to_string(index=False))

    # print('----------------------------------------------------------------')
    # print('MatPower validation')

    # # Restore line data
    # mpc['branch'] = previous_branch_array

    # # Run OPF
    # mpopt = m.mpoption('verbose', 2)
    # [baseMVA, bus, gen, gencost, branch, f, success, et] = m.runopf(mpc, mpopt, nout='max_nout')

    # mat_gen_index = range(1,len(gen)+1)
    # mat_gen_info_columns = ['bus','Pg','Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q',	'apf','unknown1','unknown2','unknown3','unknown4']
    # mat_gen_info = pd.DataFrame(gen,index = mat_gen_index, columns = mat_gen_info_columns)

    # matpower_gen_mw_total = mat_gen_info['Pg'].sum() 

    # print('----------------------------------------------------------------')
    # print('Matpower total gen MW:', matpower_gen_mw_total)
    # print('----------------------------------------------------------------')
    # print('Difference total gen MW:', P_total - (matpower_gen_mw_total))
    # P_loss_total = 0

    # for line in Line_info.index:
        
    #     if instance.P_line_loss[line].expr() >= 1e-4:
    #         ploss = instance.P_line_loss[line].expr()
    #     else:
    #         ploss = 0
    #     P_loss_total = P_loss_total + ploss
    #     #print(f"{bus}-Bus Generation: {pgen}MW")
    # print(f"Total P loss: {P_loss_total}MW")

    # """
    # Export result file
    # - Variable
    # - Dual Variable (경우에 따라 출력되지 않는 경우도 존재함)
    # """
    # ## List for storing variable dataframe
    # var_df_list = []

    # ## Variables
    # var_idx = 0
    # for mv in instance.component_objects(ctype=pyo.Var):
    #     if mv.dim() == 1: # Index dimension == 1
    #         var_columns = ['Variable_name','Index: '+mv.index_set().name, 'Value']
    #         max_var_dim = 1
    #     else: # Index dimension >= 1
    #         var_columns = ['Variable_name']
            
    #         subsets_list = list(mv.index_set().domain.subsets())
    #         for d in subsets_list:
    #             var_columns.append('Index: '+d.name)
            
    #         var_columns.append('Value')
            
    #     var_index = mv.index_set().ordered_data()
        
    #     if mv.name == 'V_ang': # Voltage angle
    #         var_df = pd.DataFrame(index = var_index, columns = var_columns)
    #         var_deg_df= pd.DataFrame(index = var_index, columns = var_columns)
    #         for idx in var_index:
    #             var_df.loc[idx,var_columns[0]] = mv.name
    #             var_deg_df.loc[idx,var_columns[0]] = 'V_ang(Deg)'
                
    #             var_df.loc[idx,var_columns[1]] = idx
    #             var_deg_df.loc[idx,var_columns[1]] = idx
    #             var_df.loc[idx,var_columns[2]] = mv[idx].value
    #             var_deg_df.loc[idx,var_columns[2]] = mv[idx].value * 180 / np.pi  # Radian to degree
        
    #     else:    
    #         var_df = pd.DataFrame(index = var_index, columns = var_columns)
    #         for idx in var_index:
    #             var_df.loc[idx,var_columns[0]] = mv.name
    #             if mv.dim() == 1:
    #                 var_df.loc[idx,var_columns[1]] = idx
    #                 var_df.loc[idx,var_columns[2]] = mv[idx].value
    #             else:
    #                 for d in range(0,mv.dim()):
    #                     var_df.loc[idx,var_columns[d+1]] = idx[d]
                        
    #                 var_df.loc[idx,var_columns[mv.dim()+1]] = mv[idx].value
        
    #     if mv.name == 'V_ang': # Voltage angle
    #         var_df_list.append(var_df)
    #         var_df_list.append(var_deg_df)
    #     else:
    #         var_df_list.append(var_df)
        
    #     var_idx+=1
        
    # ## Expressions
    # expr_idx = 0
    # for me in instance.component_objects(ctype=pyo.Expression):
    #     if me.dim() == 1: # Index dimension == 1
    #         var_columns = ['Variable_name','Index: '+me.index_set().name, 'Value']
    #         max_var_dim = 1
    #     else: # Index dimension >= 1
    #         var_columns = ['Variable_name']
            
    #         subsets_list = list(me.index_set().domain.subsets())
    #         for d in subsets_list:
    #             var_columns.append('Index: '+d.name)    
                
    #         var_columns.append('Value')
    #         max_var_dim = me.dim()
        
    #     var_index = me.index_set().ordered_data()
        
    #     var_df = pd.DataFrame(index = var_index, columns = var_columns)
    #     for idx in var_index:
    #         var_df.loc[idx,var_columns[0]] = me.name
    #         if me.dim() == 1:
    #             var_df.loc[idx,var_columns[1]] = idx
    #             var_df.loc[idx,var_columns[2]] = me[idx].expr()
    #         else:
    #             for d in range(0,me.dim()):
    #                 var_df.loc[idx,var_columns[d+1]] = idx[d]
                    
    #             var_df.loc[idx,var_columns[me.dim()+1]] = me[idx].expr()

    #     var_df_list.append(var_df)
    #     expr_idx +=1

    # ## Variables and Expression name list
    # var_n_expr_column = ['Name', 'Variable', 'Expression']
    # var_n_expr_list_df = pd.DataFrame(index = range(0,var_idx+expr_idx+1),columns=var_n_expr_column)
    # df_idx = 0
    # for df in var_df_list:
    #     if df_idx <= var_idx: # Variable list
    #         var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
    #         var_n_expr_list_df.loc[df_idx,'Variable'] = 1
    #         var_n_expr_list_df.loc[df_idx,'Expression'] = 0
    #     else: # Expression list
    #         var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
    #         var_n_expr_list_df.loc[df_idx,'Variable'] = 0
    #         var_n_expr_list_df.loc[df_idx,'Expression'] = 1
    #     df_idx += 1

    # var_df_list.insert(0,var_n_expr_list_df)

    # ## List for storing dual variable dataframe
    # dual_var_df_list = []

    # ## Dual Variables
    # try:
    #     for c in instance.component_objects(pyo.Constraint, active=True):
            
    #         if c.dim() == 1: # Index dimension == 1
    #             var_columns = ['Constraint_name','Index: '+c.index_set().name, 'Value']
    #             max_var_dim = 1
    #         else: # Index dimension >= 1
    #             var_columns = ['Constraint_name']
                
    #             subsets_list = list(c.index_set().domain.subsets())
    #             for d in subsets_list:
    #                 var_columns.append('Index: '+d.name)
                
    #             var_columns.append('Value')

    #         var_index = c.index_set().ordered_data()
    #         var_df = pd.DataFrame(index = var_index, columns = var_columns)
    #         for idx in c:
    #             var_df.loc[idx,var_columns[0]] = c.name
    #             if c.dim() == 1:
    #                 var_df.loc[idx,var_columns[1]] = idx
    #                 var_df.loc[idx,var_columns[2]] = instance.dual[c[idx]]
    #             else:
    #                 for d in range(0,c.dim()):
    #                     var_df.loc[idx,var_columns[d+1]] = idx[d]
                        
    #                 var_df.loc[idx,var_columns[c.dim()+1]] = instance.dual[c[idx]]
    #         dual_var_df_list.append(var_df)
    # except:
    #     print('Check dual')
        
    # ## Write excel
    # with pd.ExcelWriter(output_directory+'Variables/'+ simul_case +'Variables.xlsx') as writer:  
    #     for df in var_df_list:
    #         try:
    #             df.to_excel(writer, sheet_name=df['Variable_name'].values[0],index=False)
    #         except:
    #             df.to_excel(writer, sheet_name='Variable_list',index=False)

    # try:
    #     with pd.ExcelWriter(output_directory+'Dual/'+ simul_case +'Dual_Variables.xlsx') as writer:  
    #         for df in dual_var_df_list:
    #             try:
    #                 df.to_excel(writer, sheet_name=df['Constraint_name'].values[0],index=False)
    #             except:
    #                 df.to_excel(writer, sheet_name='Constraint_list',index=False)
    # except:
    #     print('Check Dual')

    # print("solve done!")






