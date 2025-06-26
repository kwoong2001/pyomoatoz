"""
MPOPF_Case_33bw_without_switch_and_matpower
- Matpower 에 기반한 MPOPF 구현
- 선로 switching이 없는 버젼

"""

import pandas as pd
import numpy as np
import os, sys
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

# Set time
T = 24

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')

simul_case = '33bus_MPOPF_problem_'

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

# # Save the whole line data
previous_branch_array = branches.copy()
    
# Set values and parameters (Bus, Line, Gen, Load, Ymatrix, Time)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info, Time_info]=Set_All_Values(np,pd,save_directory,m,mpc,previous_branch_array, T)

"""
Create OPF model and Run Pyomo
"""

# OPF Model Create
model = OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info)

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
    
    if mv.name == 'V_ang': # Voltage angle
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        var_deg_df= pd.DataFrame(index = var_index, columns = var_columns)
        for idx in var_index:
            var_df.loc[idx,var_columns[0]] = mv.name
            var_deg_df.loc[idx,var_columns[0]] = 'V_ang(Deg)'
            
            var_df.loc[idx,var_columns[1]] = idx
            var_deg_df.loc[idx,var_columns[1]] = idx
            var_df.loc[idx,var_columns[2]] = mv[idx].value
            var_deg_df.loc[idx,var_columns[2]] = mv[idx].value * 180 / np.pi  # Radian to degree
    
    else:    
        var_df = pd.DataFrame(index = var_index, columns = var_columns)
        for idx in var_index:
            var_df.loc[idx,var_columns[0]] = mv.name
            if mv.dim() == 1:
                var_df.loc[idx,var_columns[1]] = idx
                var_df.loc[idx,var_columns[2]] = mv[idx].value
            else:
                for d in range(0,mv.dim()):
                    var_df.loc[idx,var_columns[d+1]] = idx[d]
                    
                var_df.loc[idx,var_columns[mv.dim()+1]] = mv[idx].value
    
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