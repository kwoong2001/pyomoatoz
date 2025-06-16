import numpy as np
import pandas as pd
import os, sys
import pyomo.environ as pyo
from pathlib import Path
from Packages.OPF_settings import *
from Packages.System_value import *
from pyomo import environ as pym
from matpower import start_instance
from oct2py import octave


# Set directory
save_directory = Path(__file__).parent / "Pre_cal_data"      # Set save parameter directory
output_directory = Path(__file__).parent / "Output_data"    # Set output directory

save_directory.mkdir(exist_ok=True)
output_directory.mkdir(exist_ok=True)

# Set and load Matpower case
m = start_instance()
mpc = m.loadcase('case33bw')

simul_case = '33bus_NLP_Opt_problem_'

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
previous_branch_array = branches.copy()  # Save disconnected lines data

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info] = Set_All_Values(np, pd, save_directory, m, mpc, previous_branch_array)

# OPF Model Create
model = OPF_model_without_switch(np, pyo, base_MVA, Slackbus, Bus_info, Line_info, Load_info, Gen_info)

os.chdir(save_directory)
instance = model.create_instance(str(save_directory / 'Model_data.dat'))
os.chdir(os.path.dirname(__file__))

print('initialization OPF model')

optimizer = pyo.SolverFactory('ipopt')

Problem = optimizer.solve(instance, tee=True)
print('solving OPF model')

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

P_loss_total = 0

for line in Line_info.index:
    
    if instance.P_line_loss[line].expr() >= 1e-4:
        ploss = instance.P_line_loss[line].expr()
    else:
        ploss = 0
    P_loss_total = P_loss_total + ploss
    #print(f"{bus}-Bus Generation: {pgen}MW")
print(f"Total P loss: {P_loss_total}MW")