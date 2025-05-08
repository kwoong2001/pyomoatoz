"""
OPF_Case_33bw_with_switch_and_matpower
- Pandapower가 아닌 Matpower 에 기반한 OPF 구현
- 69 모선까지는 결과 도출 가능, 118 모선 이상에서는 수정 필요

"""

import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
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

simul_case = '33bus_MINLP_Opt_problem_'

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
branch_idx = 1
for branch in branches:
    if branch[-3] == 0:
        branch[-3] = 1
        print(f"{branch_idx}-th line(from bus:{branch[0]}, to bus:{branch[1]}) disconnected --> connected")
    branch_idx +=1
    
# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info]=Set_All_Values_with_switch(np,pd,save_directory,m,mpc,previous_branch_array)

"""
Create OPF model and Run Pyomo
"""

# OPF Model Create
model = OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info)

# Create instance for OPF Model
os.chdir(save_directory)
instance = model.create_instance(save_directory + 'Model_data.dat')
os.chdir(os.path.dirname(__file__))

print('Initializing OPF model...')

"""
NEOS 기반 Solver 활용 - 가입 필요 (무료)
https://neos-server.org/neos/

# formulate optimization model with NEOS
os.environ['NEOS_EMAIL'] = ''
optimizer = pyo.SolverManagerFactory('neos')
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Problem = optimizer.solve(instance, opt='knitro')

"""

optimizer = pyo.SolverFactory('mindtpy')

instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
Problem = optimizer.solve(instance,mip_solver="glpk", nlp_solver="ipopt",tee=True)
print('Solving OPF model...')


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
print('MatPower validation')

#Restore line data
#mpc['branch'] = previous_branch_array

# Run OPF
mpopt = m.mpoption('verbose', 2)
[baseMVA, bus, gen, gencost, branch, f, success, et] = m.runopf(mpc, mpopt, nout='max_nout')

mat_gen_index = range(1,len(gen)+1)
mat_gen_info_columns = ['bus','Pg',	'Qg','Qmax','Qmin','Vg','mBase','status','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max','Qc2min','Qc2max','ramp_agc','ramp_10','ramp_30','ramp_q',	'apf','unknown1','unknown2','unknown3','unknown4']
mat_gen_info = pd.DataFrame(gen,index = mat_gen_index, columns = mat_gen_info_columns)

matpower_gen_mw_total = mat_gen_info['Pg'].sum() 

print('----------------------------------------------------------------')
print('Matpower total gen MW:', matpower_gen_mw_total)
print('----------------------------------------------------------------')
print('Difference total gen MW:', P_total - (matpower_gen_mw_total))
P_loss_total = 0

for line in Line_info.index:
    
    if instance.P_line_loss[line].expr() >= 1e-4:
        ploss = instance.P_line_loss[line].expr()
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
try:
    for c in instance.component_objects(pyo.Constraint, active=True):
        var_columns = ['Constraint_name','Index: '+c.index_set().name, 'Value']
        var_index = c.index_set()
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        for index in c:
            var_df.loc[index,var_columns[0]]=c.name
            var_df.loc[index,var_columns[1]]=index
            var_df.loc[index,var_columns[2]]=instance.dual[c[index]]
        dual_var_df_list.append(var_df)
except:
    print('Check dual')
    
## Write excel
with pd.ExcelWriter(output_directory+'Variables/'+ simul_case +'Variables.xlsx') as writer:  
    for df in var_df_list:
        try:
            df.to_excel(writer, sheet_name=df.loc[1,'Variable_name'],index=False)
        except:
            df.to_excel(writer, sheet_name='Variable_list',index=False)

try:
    with pd.ExcelWriter(output_directory+'Dual/'+ simul_case +'Dual_Variables.xlsx') as writer:  
        for df in dual_var_df_list:
            try:
                df.to_excel(writer, sheet_name=df.loc[1,'Constraint_name'],index=False)
            except:
                df.to_excel(writer, sheet_name='Constraint_list',index=False)
except:
    print('Check Dual')

print("solve done!")