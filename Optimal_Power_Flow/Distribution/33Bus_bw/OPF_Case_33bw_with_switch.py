"""
OPF_Case_33bw_with_switch
- OPF_Case_33bw_pyomo 250425_V4 기반

250501_V5: In service 'False'로 된 선로들 모두 'True'로 바꾸어서 Y 행렬 구성, 선로 상태에 따라 G와 B 값이 최적화 과정에서 바뀔 수 있도록 구성 필요
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from Packages.Set_values import *
from Packages.OPF_Creator import *

"""
Set model and parameters with Pandapower
"""

## Set directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"       # Set save parameter directory
output_directory = os.path.dirname(__file__) + "/Output_data/"     # Set output directory

# Set and load Pandapower Case
net = pn.case33bw()

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

"""
Export result file
"""

for bus in Bus_info.index:
    
    if instance.V_mag[bus].value >= 1e-4:
        vmag = instance.V_mag[bus].value
    else:
        vmag = 0
    
    if np.absolute(instance.V_ang[bus].value) >= 1e-4:
        vang = instance.V_ang[bus].value * 180 / np.pi
    else:
        vang = 0
    
    #print(f"{bus}-Bus Voltage magnitude: {vmag:.4f} [pu], angle: {vang:.2f} [deg]")

print("Dual Variables - Base value: Node Margianl Price(20) * base_MVA (10)")
for c in instance.component_objects(pyo.Constraint, active=True):
    if str(c) == 'P_bal_con':
        print("   Constraint", c)
        for index in c:
            print("      ", index, instance.dual[c[index]])

for c in instance.component_objects(pyo.Constraint, active=True):
    if str(c) == 'P_gen_min_con':
        print("   Constraint", c)
        for index in c:
            print("      ", index, instance.dual[c[index]])
            
for l in Line_info.index:
    p_line_loss = instance.P_line_loss[l].expr()* base_MVA
    q_line_loss = instance.Q_line_loss[l].expr()* base_MVA
    base_current = base_MVA /Bus_info['vn_kv'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
    I_line = instance.I_line_mag[l].expr()
    I_line_loading_percent = instance.I_loading_percent[l].expr()
    print(f"{l}-line current: {I_line:.5f} [A], loading percent: {I_line_loading_percent} [%]")
    #print(f"{l}-line flow (receive) reactive power: {q_line:.2f} [MVar]")

for c in instance.component_objects(pyo.Var, active=True):
    if str(c) == 'P_bal_con':
        print("   Constraint", c)
        for index in c:
            print("      ", index, instance.dual[c[index]])     

print("solve done!")