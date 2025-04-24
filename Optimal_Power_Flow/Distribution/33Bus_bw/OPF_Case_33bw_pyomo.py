"""
OPF_Case_33bw
250424_V2: PU 단위의 OPF 문제를 정식화하고 결과를 도출하는 코드 - 33Bus 해 찾기 성공

250422_V1: Pandapower로 부터 불러온 계통 모델을 기반으로 Pyomo에 최적화 문제를 해결할 수 있도록 구성
"""

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from Packages.Set_values import *
from Packages.OPF_Creator import *

# Set save parameter directory
save_directory = os.path.dirname(__file__) + "/Pre_cal_data/"

# Load Pandapower Case
net = pn.case33bw()

# Set slack bus
Slackbus = 1

# Run loadflow
pp.runpp(net,numba=False)
base_MVA = net._ppc['baseMVA'] #Base MVA

# Set values and parameters (Bus, Line, Gen, Load, Ymatrix)
[Bus_info, Line_info, Gen_info, Load_info, Y_mat_info]=Set_All_Values(np,pd,save_directory,net)

# OPF Model Create
model = OPF_model_creator(pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info)

# Create instance for OPF Model
os.chdir(save_directory)
instance = model.create_instance(save_directory + 'Model_data.dat')
os.chdir(os.path.dirname(__file__))

print('Initializing OPF model...')
optimizer = pyo.SolverFactory('ipopt') 
optimizer.options["max_iter"] = 100000

print('Solving OPF model...')
Problem = optimizer.solve(instance,tee=True)

for bus in Bus_info.index:
    
    if instance.V_mag[bus].value >= 1e-4:
        vmag = instance.V_mag[bus].value
    else:
        vmag = 0
    
    if np.absolute(instance.V_ang[bus].value) >= 1e-4:
        vang = instance.V_ang[bus].value * 180 / np.pi
    else:
        vang = 0
    
    print(f"{bus}-Bus Voltage magnitude: {vmag:.4f} [pu], angle: {vang:.2f} [deg]")


for bus in Bus_info.index:
    
    if instance.PGen[bus].value >= 1e-4:
        pgen = instance.PGen[bus].value * base_MVA
    else:
        pgen = 0
    print(f"{bus}-Bus Generation: {pgen}MW")
    
     

print("solve done!")