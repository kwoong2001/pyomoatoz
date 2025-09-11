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
from collections import defaultdict, deque, Counter
from config import *

"""
Set model and parameters with Matpower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')


if dg_case == 'none':
    if switch == 1:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_' + dg_case + '_dgs_' + str(Ta) + '_interval_'
    elif switch == 0:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_' + dg_case + '_dgs_'
else:
    if switch == 1:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_with_switch_' + dg_case + '_dgs_' + str(pv_penetration) + '_pv_penetration_' + str(Ta) + '_interval_'
    elif switch == 0:
        simul_case = '33bus_MINLP_Opt_problem_for_min_cost_without_switch_' + dg_case + '_dgs_' + str(pv_penetration) + '_pv_penetration_'

print(simul_case)

# Base MVA, Bus, Branch, Generators
base_MVA = mpc['baseMVA']
    
# Find slack bus, add distributed generators, set branch status
[Slackbus, previous_branch_array, pv_curtailment_df] = Set_System_Env(np,pd,save_directory,mpc,switch,dg_case,pv_penetration)

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix, Time)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info, Time_info, Time_interval_info]=Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array,T,Ta,Tp)

print(Gen_info)

# Set profiles of distributed generators and load
[DG_profile_df, Load_profile_df] = Set_Resource_Profiles(np,pd,save_directory,T,Load_info,dg_case,pv_penetration)

"""
Create OPF model and Run Pyomo
"""
# OPF Model Create
if switch == 1:
    model = OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,Time_interval_info,Ta,Tp,DG_profile_df,Load_profile_df)
elif switch == 0:
    model = OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,DG_profile_df,Load_profile_df)

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
# Create Log directory if it doesn't exist
log_dir = os.path.join(output_directory, "Log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, simul_case + ".log")

Problem = optimizer.solve(instance, tee=True, logfile=log_path)

print('Solving OPF model...')


"""
Result
"""

print('----------------------------------------------------------------')
Objective_value = instance.obj()
print(f'Objective value = {Objective_value: .4f}')
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

    # 결과값(result) 시트 저장
    opf_result = pd.DataFrame({
    'Name': ['Objective_value', 'P_total', 'D_total', 'P_loss_total'],
    'Value': [Objective_value, P_total, D_total, P_loss_total]
    })
    opf_result.to_excel(writer, sheet_name='result', index=False)

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