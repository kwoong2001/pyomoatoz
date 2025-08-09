"""
OPF_Case_33_bus_with_matpower
- Matpower 에 기반한 OPF 구현
- 33, 69 bus 동작 확인
- Switching 고려한 버젼과 고려하지 않은 버젼 모두 구현 완료
"""

import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from Packages.Set_values_matpower import *
from Packages.OPF_Creator_matpower import *
from Packages.set_system_env_matpower import *
from Packages.Set_Profiles import *
from pyomo import environ as pym
from matpower import start_instance
from oct2py import octave

"""
Set model and parameters with Matpower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set time
T = 24

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')

simul_case = '33bus_MINLP_Opt_problem_for_min_cost_'

# Base MVA, Bus, Branch, Generators
base_MVA = mpc['baseMVA']
    
# Find slack bus, add distributed generators, set branch status
[Slackbus, previous_branch_array, pv_curtailment_df] = Set_System_Env(np,pd,save_directory,mpc)

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix, Time)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info, Time_info]=Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array, T)

# Set profiles of distributed generators and load
[DG_profile_df, Load_profile_df] = Set_Resource_Profiles(np,pd,save_directory,T,Load_info)

"""
Create OPF model and Run Pyomo
"""
# OPF Model Create
model = OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,DG_profile_df,Load_profile_df,pv_curtailment_df)


# Create instance for OPF Model
os.chdir(save_directory)
instance = model.create_instance(save_directory + 'Model_data.dat')
os.chdir(os.path.dirname(__file__))

print('Initializing OPF model...')

#IPOPT Solver 이용
# optimizer = pyo.SolverFactory('ipopt')
# optimizer.options['max_iter'] = 30000

#KNITRO Solver 이용
optimizer = pyo.SolverFactory('knitroampl',executable='C:/Program Files/Artelys/Knitro 14.2.0/knitroampl/knitroampl.exe')
optimizer.options['mip_multistart'] = 1
optimizer.options['mip_numthreads'] = 1
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Problem = optimizer.solve(instance,tee=True,logfile="solver_logging.log")


print('Solving OPF model...')


"""
Result
"""

print('----------------------------------------------------------------')
print(f'Objective value = {instance.obj(): .4f}')
P_total = 0
D_total = 0
for bus in Bus_info.index:
    for time in Time_info['Time']:
        for gen in Gen_info.index:
            if instance.PGen[gen, bus, time].value >= 1e-4:
                pgen = instance.PGen[gen, bus, time].value * base_MVA
            else:
                pgen = 0
            P_total = P_total + pgen

        if instance.PDem[bus, time].expr() >= 1e-4:
            pdem = instance.PDem[bus, time].expr() * base_MVA
        else:
            pdem = 0
        D_total = D_total + pdem
    


print('----------------------------------------------------------------')
print('OPF Model total gen MW:', P_total)
print('OPF Model total load MW:', D_total)



# print('----------------------------------------------------------------')
# print('MatPower validation')


# #Restore line data
# #mpc['branch'] = previous_branch_array

# # Run OPF
# mpopt = m.mpoption('verbose', 2)
# [baseMVA, bus, gen, gencost, branch, f, success, et] = m.runopf(mpc, mpopt, nout='max_nout')

# mat_gen_index = range(1,len(gen)+1)
# mat_gen_info_columns = ['bus','Pg',	'Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q',	'apf','unknown1','unknown2','unknown3','unknown4']
# mat_gen_info = pd.DataFrame(gen,index = mat_gen_index, columns = mat_gen_info_columns)

# matpower_gen_mw_total = mat_gen_info['Pg'].sum() 

# print('----------------------------------------------------------------')
# print('Matpower total gen MW:', matpower_gen_mw_total)
# print('----------------------------------------------------------------')
# print('Difference total gen MW:', P_total - (matpower_gen_mw_total))
P_loss_total = 0

for line in Line_info.index:
    
    if instance.P_line_loss[line,time].expr() >= 1e-4:
        ploss = instance.P_line_loss[line,time].expr()
    else:
        ploss = 0
    P_loss_total = P_loss_total + ploss
    #print(f"{bus}-Bus Generation: {pgen}MW")
print(f"Total P loss: {P_loss_total}MW")

"""
Export result file
- Variable
- Dual Variable (경우에 따라 출력되지 않는 경우도 존재함)
"""
## List for storing variable dataframe
var_df_list = []

## Variables
var_idx = 0
for mv in instance.component_objects(ctype=pyo.Var):
    if mv.dim() == 1: # Index dimension == 1
        var_columns = ['Variable_name','Index: '+mv.index_set().name, 'Value']
        max_var_dim = 1
    else: # Index dimension >= 1
        var_columns = ['Variable_name']
        
        subsets_list = list(mv.index_set().domain.subsets())
        for d in subsets_list:
            var_columns.append('Index: '+d.name)
        
        var_columns.append('Value')
        
    var_index = mv.index_set().ordered_data()
    
    if mv.name == 'V_ang':  # Voltage angle
        var_df = pd.DataFrame(index=var_index, columns=var_columns)
        var_deg_df = pd.DataFrame(index=var_index, columns=var_columns)
        for idx in var_index:
            var_df.loc[idx, var_columns[0]] = mv.name
            var_deg_df.loc[idx, var_columns[0]] = 'V_ang(Deg)'
            # Handle multi-index
            if mv.dim() == 1:
                var_df.loc[idx, var_columns[1]] = idx
                var_deg_df.loc[idx, var_columns[1]] = idx
                var_df.loc[idx, var_columns[2]] = mv[idx].value
                var_deg_df.loc[idx, var_columns[2]] = mv[idx].value * 180 / np.pi
            else:
                for d in range(mv.dim()):
                    var_df.loc[idx, var_columns[d+1]] = idx[d]
                    var_deg_df.loc[idx, var_columns[d+1]] = idx[d]
                var_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value
                var_deg_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value * 180 / np.pi
    else:
        var_df = pd.DataFrame(index=var_index, columns=var_columns)
        for idx in var_index:
            var_df.loc[idx, var_columns[0]] = mv.name
            if mv.dim() == 1:
                var_df.loc[idx, var_columns[1]] = idx
                var_df.loc[idx, var_columns[2]] = mv[idx].value
            else:
                for d in range(mv.dim()):
                    var_df.loc[idx, var_columns[d+1]] = idx[d]
                var_df.loc[idx, var_columns[mv.dim()+1]] = mv[idx].value
    
    if mv.name == 'V_ang': # Voltage angle
        var_df_list.append(var_df)
        var_df_list.append(var_deg_df)
    else:
        var_df_list.append(var_df)
    
    var_idx+=1
    
## Expressions
expr_idx = 0
for me in instance.component_objects(ctype=pyo.Expression):
    if me.dim() == 1: # Index dimension == 1
        var_columns = ['Variable_name','Index: '+me.index_set().name, 'Value']
        max_var_dim = 1
    else: # Index dimension >= 1
        var_columns = ['Variable_name']
        
        subsets_list = list(me.index_set().domain.subsets())
        for d in subsets_list:
            var_columns.append('Index: '+d.name)    
            
        var_columns.append('Value')
        max_var_dim = me.dim()
    
    var_index = me.index_set().ordered_data()
    
    var_df = pd.DataFrame(index = var_index, columns = var_columns)
    for idx in var_index:
        var_df.loc[idx,var_columns[0]] = me.name
        if me.dim() == 1:
            var_df.loc[idx,var_columns[1]] = idx
            var_df.loc[idx,var_columns[2]] = me[idx].expr()
        else:
            for d in range(0,me.dim()):
                var_df.loc[idx,var_columns[d+1]] = idx[d]
                
            var_df.loc[idx,var_columns[me.dim()+1]] = me[idx].expr()

    var_df_list.append(var_df)
    expr_idx +=1

## Variables and Expression name list
var_n_expr_column = ['Name', 'Variable', 'Expression']
var_n_expr_list_df = pd.DataFrame(index = range(0,var_idx+expr_idx+1),columns=var_n_expr_column)
df_idx = 0
for df in var_df_list:
    if df_idx <= var_idx: # Variable list
        var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
        var_n_expr_list_df.loc[df_idx,'Variable'] = 1
        var_n_expr_list_df.loc[df_idx,'Expression'] = 0
    else: # Expression list
        var_n_expr_list_df.loc[df_idx,'Name'] = df['Variable_name'].values[0]
        var_n_expr_list_df.loc[df_idx,'Variable'] = 0
        var_n_expr_list_df.loc[df_idx,'Expression'] = 1
    df_idx += 1

var_df_list.insert(0,var_n_expr_list_df)

## List for storing dual variable dataframe
dual_var_df_list = []

## Dual Variables
try:
    for c in instance.component_objects(pyo.Constraint, active=True):
        
        if c.dim() == 1: # Index dimension == 1
            var_columns = ['Constraint_name','Index: '+c.index_set().name, 'Value']
            max_var_dim = 1
        else: # Index dimension >= 1
            var_columns = ['Constraint_name']
            
            subsets_list = list(c.index_set().domain.subsets())
            for d in subsets_list:
                var_columns.append('Index: '+d.name)
            
            var_columns.append('Value')

        var_index = c.index_set().ordered_data()
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        for idx in c:
            var_df.loc[idx,var_columns[0]] = c.name
            if c.dim() == 1:
                var_df.loc[idx,var_columns[1]] = idx
                var_df.loc[idx,var_columns[2]] = instance.dual[c[idx]]
            else:
                for d in range(0,c.dim()):
                    var_df.loc[idx,var_columns[d+1]] = idx[d]
                    
                var_df.loc[idx,var_columns[c.dim()+1]] = instance.dual[c[idx]]
        dual_var_df_list.append(var_df)
except:
    print('Check dual')
    
## Write excel
with pd.ExcelWriter(output_directory+'Variables/'+ simul_case +'Variables.xlsx') as writer:  
    for df in var_df_list:
        try:
            df.to_excel(writer, sheet_name=df['Variable_name'].values[0],index=False)
        except:
            df.to_excel(writer, sheet_name='Variable_list',index=False)

try:
    with pd.ExcelWriter(output_directory+'Dual/'+ simul_case +'Dual_Variables.xlsx') as writer:  
        for df in dual_var_df_list:
            try:
                df.to_excel(writer, sheet_name=df['Constraint_name'].values[0],index=False)
            except:
                df.to_excel(writer, sheet_name='Constraint_list',index=False)
except:
    print('Check Dual')

print("solve done!")

# DG list
dg_list = []

list_dg = pd.read_excel(save_directory + 'DG_Candidates.xlsx', sheet_name='Candidate')
list_dg_df = pd.DataFrame(list_dg, columns=['Index','Bus number', 'Rating[MW]', 'Type', 'Profile'])
print(list_dg_df)

# Plotting the network
excel_path = output_directory + 'Variables/' + simul_case + 'Variables.xlsx'
line_status_df = pd.read_excel(excel_path, sheet_name='Line_Status')

# line_status_df의 'Index: Lines'를 int로 변환 (혹시 타입이 다를 경우)
line_status_df['Index: Lines'] = line_status_df['Index: Lines'].astype(int)


# merge로 합치기 (left_on: Line_l, right_on: Index: Lines)
merged_df = pd.merge(Line_info, line_status_df, left_on='Line_l', right_on='Index: Lines', how='left')

# 필요한 컬럼만 추출 (예시: line index, from_bus, to_bus, line status)
result_df = merged_df[['Index: Lines', 'from_bus', 'to_bus', 'Value']].copy()
result_df = result_df.rename(columns={'Index: Lines': 'line_index', 'Value': 'line_status'})

result_df.to_excel(os.path.join(output_directory, "result_df.xlsx"), index=False)
print(result_df)

line_pairs = []
for i in range(len(line_status_df)):
    if line_status_df.loc[i, 'Value'] == 1:
        line_idx = line_status_df.loc[i, 'Index: Lines']
        from_bus = int(Line_info.loc[line_idx, 'from_bus'])
        to_bus = int(Line_info.loc[line_idx, 'to_bus'])
        line_pairs.append((from_bus, to_bus))

# 버스 위치 정보 불러오기 (sheet 이름 명시)
bus_pos_df = pd.read_excel(os.path.join(save_directory, "System.xlsx"), sheet_name="33bus", usecols=["Bus", "x", "y"])

pos = {}
for i in range(len(bus_pos_df)):
    bus = int(bus_pos_df.loc[i, "Bus"])
    x = bus_pos_df.loc[i, "x"]
    y = bus_pos_df.loc[i, "y"]
    pos[bus] = (x, y)
# 선로(from, to)
branches = line_pairs

fig, ax = plt.subplots(figsize=(12, 10))


for i, (from_bus, to_bus) in enumerate(branches):
    from_x, from_y = pos[from_bus]
    to_x, to_y = pos[to_bus]

    # y좌표가 같고 x좌표 차이가 2 이상인 경우
    if from_y == to_y and abs(from_x - to_x) >= 2:
        mid_y = from_y + 0.4 + 0.1 * (i % 5)  # 선로마다 살짝 다르게
        # 출발점에서 위로, 도착점에서 위로, 위에서 수평 연결
        ax.plot([from_x, from_x], [from_y, mid_y], color='black', linewidth=1)
        ax.plot([to_x, to_x], [to_y, mid_y], color='black', linewidth=1)
        ax.plot([from_x, to_x], [mid_y, mid_y], color='black', linewidth=1)
    # y좌표가 다르고 차이가 2 이상인 경우
    elif abs(from_y - to_y) >= 2:
        mid_y = (from_y + to_y) / 2 + 0.1 * (i % 5) - 0.4  # 중간에서 살짝 위로
        # 출발점에서 중간까지, 도착점에서 중간까지, 중간끼리 수평 연결
        ax.plot([from_x, from_x], [from_y, mid_y], color='black', linewidth=1)
        ax.plot([to_x, to_x], [to_y, mid_y], color='black', linewidth=1)
        ax.plot([from_x, to_x], [mid_y, mid_y], color='black', linewidth=1)
    else:
        # 그냥 직선 연결
        ax.plot([from_x, to_x], [from_y, to_y], color='black', linewidth=1)

# 모선 (점) 그리기
for bus, (x, y) in pos.items():
    ax.plot(x, y, 'o', color='black', markersize=5)  # 점으로 표시
    ax.text(x + 0.25, y - 0.3, str(bus), ha='center', va='top')

ax.set_aspect('equal')
ax.axis('off')

fig_path = os.path.join(output_directory, "33bus_network_re.png")
plt.savefig(fig_path, bbox_inches='tight', dpi=300)