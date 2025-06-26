"""
OPF_Case_33bw
250508_V5: 결과를 담을 수 있는 엑셀 파일 등 구성

250425_V4: 선로 rating 제약조건 반영, Sending and receiving power and current 반영, Slack 검출 코드 반영

250424_V3: 발전 비용을 반영한 OPF 문제 구성 및 Dual Variable 추출

250424_V2: PU 단위의 OPF 문제를 정식화하고 결과를 도출하는 코드 - 33Bus 해 찾기 성공

250422_V1: Pandapower로 부터 불러온 계통 모델을 기반으로 Pyomo에 최적화 문제를 해결할 수 있도록 구성
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from Packages.Set_values_pandapower import *
from Packages.OPF_Creator_pandapower import *

"""
Set model and parameters with Pandapower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set and load Pandapower Case
net = pn.case33bw()

simul_case = '33bus_NLP_Opt_problem_'


"""
Run loadflow and load system data
"""

# Run loadflow
pp.runpp(net,numba=False)
base_MVA = net._ppc['baseMVA'] #Base MVA

# Find and load slack bus
Slackbus = 0
if net._ppc['bus'][0][0] == 0:
    compensation_idx_factor = 1
else:
    compensation_idx_factor = 0

for busidx in range(net._ppc['bus'].shape[0]):
    if net._ppc['bus'][busidx][1] == 3: 
        Slackbus = int(net._ppc['bus'][busidx][0]) + compensation_idx_factor
        
print(f"{net.bus.shape[0]}-buses case, Slack bus: [{Slackbus}]")

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info]=Set_All_Values(np,pd,save_directory,net)

"""
Create OPF model and Run Pyomo
"""

# OPF Model Create
model = OPF_model_creator(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info)

# Create instance for OPF Model
os.chdir(save_directory)
instance = model.create_instance(save_directory + 'Model_data.dat')
os.chdir(os.path.dirname(__file__))

print('Initializing OPF model...')
optimizer = pyo.SolverFactory('ipopt') 
optimizer.options["max_iter"] = 100000

print('Solving OPF model...')
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Problem = optimizer.solve(instance,tee=True)

"""
Result
"""

print('----------------------------------------------------------------')
print(f'Objective value = {instance.obj(): .4f}')

P_total = 0
D_total = 0
for bus in Bus_info.index:
    
    if instance.PGen[bus].value >= 1e-4:
        pgen = instance.PGen[bus].value * base_MVA
    else:
        pgen = 0
    P_total = P_total + pgen
    #print(f"{bus}-Bus Generation: {pgen}MW")
    
    if instance.PDem[bus].expr()>=1e-4:
        pdem = instance.PDem[bus].expr() * base_MVA
    else:
        pdem = 0
    D_total = D_total + pdem

print('----------------------------------------------------------------')
print('OPF Model total gen MW:', P_total)
print('OPF Model total load MW:', D_total)

print('----------------------------------------------------------------')
print('Pandapower validation')
pp.runopp(net, delta=1e-16,numba=False)

panda_gen_mw_total = net.res_gen['p_mw'].sum() 
panda_imports_mw_total = net.res_ext_grid['p_mw'].sum()

print('----------------------------------------------------------------')
print('Panda total gen MW:', panda_gen_mw_total + panda_imports_mw_total)
print('Panda total load MW:', net.res_load['p_mw'].sum())
print('----------------------------------------------------------------')
print('Difference total gen MW:', P_total - (panda_gen_mw_total + panda_imports_mw_total))
print('Difference total load MW:', D_total - (net.res_load['p_mw'].sum()))

P_loss_total = 0

for line in Line_info.index:
    
    if instance.P_line_loss[line].expr() >= 1e-4:
        ploss = instance.P_line_loss[line].expr()
    else:
        ploss = 0
    P_loss_total = P_loss_total + ploss
    
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
    var_columns = ['Variable_name','Index: '+mv.index_set().name, 'Value']
    var_index = mv.index_set()
    if mv.name == 'V_ang': # Voltage angle
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        var_deg_df= pd.DataFrame(index = var_index, columns = var_columns)
        for idx in mv.index_set():
            var_df.loc[idx,var_columns[0]] = mv.name
            var_deg_df.loc[idx,var_columns[0]] = 'V_ang(Deg)'
            
            var_df.loc[idx,var_columns[1]] = idx
            var_deg_df.loc[idx,var_columns[1]] = idx
            var_df.loc[idx,var_columns[2]] = mv[idx].value
            var_deg_df.loc[idx,var_columns[2]] = mv[idx].value * 180 / np.pi  # Radian to degree
    
    else:    
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        for idx in mv.index_set():
            var_df.loc[idx,var_columns[0]] = mv.name
            var_df.loc[idx,var_columns[1]] = idx
            var_df.loc[idx,var_columns[2]] = mv[idx].value
    
    if mv.name == 'V_ang': # Voltage angle
        var_df_list.append(var_df)
        var_df_list.append(var_deg_df)
    else:
        var_df_list.append(var_df)
    
    var_idx+=1
    
## Expressions
expr_idx = 0
for me in instance.component_objects(ctype=pyo.Expression):
    var_columns = ['Variable_name','Index: '+me.index_set().name, 'Value']
    var_index = me.index_set()
    var_df = pd.DataFrame(index = var_index, columns = var_columns)
    for idx in me.index_set():
        var_df.loc[idx,var_columns[0]] = me.name
        var_df.loc[idx,var_columns[1]] = idx
        var_df.loc[idx,var_columns[2]] = me[idx].expr()
    
    var_df_list.append(var_df)
    expr_idx +=1

## Variables and Expression name list
var_n_expr_column = ['Name', 'Variable', 'Expression']
var_n_expr_list_df = pd.DataFrame(index = range(0,var_idx+expr_idx+1),columns=var_n_expr_column)
df_idx = 0
for df in var_df_list:
    if df_idx <= var_idx:
        var_n_expr_list_df.loc[df_idx,'Name'] = df.loc[1,'Variable_name']
        var_n_expr_list_df.loc[df_idx,'Variable'] = 1
        var_n_expr_list_df.loc[df_idx,'Expression'] = 0
    else:
        var_n_expr_list_df.loc[df_idx,'Name'] = df.loc[1,'Variable_name']
        var_n_expr_list_df.loc[df_idx,'Variable'] = 0
        var_n_expr_list_df.loc[df_idx,'Expression'] = 1
    df_idx += 1

var_df_list.insert(0,var_n_expr_list_df)

## List for storing dual variable dataframe
dual_var_df_list = []

## Dual Variables
for c in instance.component_objects(pyo.Constraint, active=True):
    var_columns = ['Constraint_name','Index: '+c.index_set().name, 'Value']
    var_index = c.index_set()
    var_df = pd.DataFrame(index = var_index, columns = var_columns)
    for index in c:
        var_df.loc[index,var_columns[0]]=c.name
        var_df.loc[index,var_columns[1]]=index
        var_df.loc[index,var_columns[2]]=instance.dual[c[index]]
    dual_var_df_list.append(var_df)
    
## Write excel
with pd.ExcelWriter(output_directory+'Variables/'+ simul_case +'Variables.xlsx') as writer:  
    for df in var_df_list:
        try:
            df.to_excel(writer, sheet_name=df.loc[1,'Variable_name'],index=False)
        except:
            df.to_excel(writer, sheet_name='Variable_list',index=False)

with pd.ExcelWriter(output_directory+'Dual/'+ simul_case +'Dual_Variables.xlsx') as writer:  
    for df in dual_var_df_list:
        try:
            df.to_excel(writer, sheet_name=df.loc[1,'Constraint_name'],index=False)
        except:
            df.to_excel(writer, sheet_name='Constraint_list',index=False)

print("solve done!")