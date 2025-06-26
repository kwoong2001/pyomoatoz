"""
OPF_Case_33bw_with_switch
- OPF_Case_33bw_pyomo 250425_V4 기반

250508_V5: MINLP 문제를 해결할 수 있도록 구성하였으며, NEOS 사용할 수 있도록 구성
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from Packages.Set_values_pandapower import *
from Packages.OPF_Creator_pandapower import *
from pyomo import environ as pym
   

"""
Set model and parameters with Pandapower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set and load Pandapower Case
net = pn.case33bw()

simul_case = '33bus_MINLP_Opt_problem_'

compensation_idx_factor = 0 # Pandapower의 bus index가 0부터 시작하는 부분을 보정하기 위한 상수 (bus index가 0이면 1을 더하여 bus가 1부터 시작하도록 만들어줌)
if net.bus.loc[0,'name'] == 0:
    compensation_idx_factor = 1
else:
    compensation_idx_factor = 0
    
# Change disconnected line to connected line
previous_line_df = net.line.copy() # Save disconnected lines data
for line in net.line.index:
    if net.line.loc[line,'in_service'] == False:
        net.line.loc[line,'in_service'] = True
        print(f"{line+1}-th line(from bus:{net.line.loc[line,'from_bus']+compensation_idx_factor}, to bus:{net.line.loc[line,'to_bus']+compensation_idx_factor}) disconnected --> connected")
    
"""
Run loadflow and load system data
"""

# Run loadflow
pp.runpp(net,numba=False)
base_MVA = net._ppc['baseMVA'] #Base MVA

# Find and load slack bus
Slackbus = 0
compensation_idx_factor = 0 # Pandapower의 bus index가 0부터 시작하는 부분을 보정하기 위한 상수 (bus index가 0이면 1을 더하여 bus가 1부터 시작하도록 만들어줌)
if net._ppc['bus'][0][0] == 0:
    compensation_idx_factor = 1 
else:
    compensation_idx_factor = 0

for busidx in range(net._ppc['bus'].shape[0]):
    if net._ppc['bus'][busidx][1] == 3: 
        Slackbus = int(net._ppc['bus'][busidx][0]) + compensation_idx_factor
        
print(f"{net.bus.shape[0]}-buses case, Slack bus: [{Slackbus}]")

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info]=Set_All_Values_with_switch(np,pd,save_directory,net,previous_line_df)

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
print('Pandapower validation')

#Restore line data
net.line = previous_line_df
pp.runpp(net,numba=False)

# Run OPF
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