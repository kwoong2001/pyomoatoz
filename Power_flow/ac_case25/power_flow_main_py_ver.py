"""
AC Power flow - 25 bus case V1 (25.04.11)
Readme_ac_powerflow.md 참고
"""

""""
A. 파라미터 및 데이터 입력
"""
case_name = 'ac_case25'
case_file_name = 'ac_case25.xlsx'

""" Call modules """
import os
import pandas as pd
import numpy as np
import pyomo.environ as pyo 
import math

""""Packages"""
from Packages.Create_Y_bus import create_Y_bus
from Packages.Create_set_n_params import create_set_and_params

""""Path"""
Main_path = os.path.dirname(os.path.realpath(__file__))
inputdata_path = Main_path + '/InputData/'
pre_caldata_path = Main_path + '/PreCalData/' # Y행렬, 전압, 유효전력 PU 초기값 등 조류계산 수행 전 계산값들이 저장되는 경로

""""Data from excel"""
power_system_data = pd.ExcelFile(inputdata_path + case_file_name)

#power_system_data.sheet_names 참고
Bus_data = pd.read_excel(power_system_data,'bus')
Branch_data = pd.read_excel(power_system_data,'branch')
Transformer_data = pd.read_excel(power_system_data,'transformer')
Gen_data = pd.read_excel(power_system_data,'generator')
Param_data = pd.read_excel(power_system_data,'param')


""""
B. 데이터 전처리 (최적화 수행에 맞는 데이터 만들기)
"""

""" 1. 집합 만들기(Set) """
create_set_and_params(np,pd,pre_caldata_path,Bus_data['Bus'])

""" 2. Y 행렬 생성 """
#[Bus_i, Bus_j, Bus_G, Bus_B]를 열로 갖는 Y_bus.csv 파일 '/PreCalData/'폴더에 생성하는 것 목표
create_Y_bus(np,pd,pre_caldata_path,Bus_data['Bus'],Branch_data,Transformer_data)

""" 3. 모선별 아는 값 입력(전압의 크기, 위상, 유효 및 무효전력) """
V_slack = 1 # Excel에서 불러오기

""""
C. 최적화 수행
"""

""" Pyomo INPUTS """
solver = 'ipopt' # Select solver

""" Pyomo Config """
# Create the abstract model - 시그마로 표현할 수 있는 모델
model = pyo.AbstractModel()

## Loaded from csv in ../PreCalData/
# Define set
model.Buses = pyo.Set(dimen=1)

# Define parameters
model.omega = pyo.Param(model.Buses,within=pyo.Any) # Slack bus indicator
model.alpha = pyo.Param(model.Buses,within=pyo.Any) # PV bus indicator
model.beta = pyo.Param(model.Buses,within=pyo.Any) # PQ bus indicator

model.P_known = pyo.Param(model.Buses,within=pyo.Any) # Known value of active power in Bus i
model.Q_known = pyo.Param(model.Buses,within=pyo.Any) # Known value of reactive power in Bus i
model.P_gen = pyo.Param(model.Buses,within=pyo.Any) # Set point of generation of active power in Bus i
model.Q_gen = pyo.Param(model.Buses,within=pyo.Any) # Set point of generation of reactive power in Bus i
model.P_load = pyo.Param(model.Buses,within=pyo.Any) # Set point of load of active power in Bus i
model.Q_load = pyo.Param(model.Buses,within=pyo.Any) # Set point of load of reactive power in Bus i
model.P_gen_max = pyo.Param(model.Buses,within=pyo.Any) # Maximum value of generation of active power in Bus i
model.P_gen_min = pyo.Param(model.Buses,within=pyo.Any) # Minimum value of generation of active power in Bus i
model.Q_gen_max = pyo.Param(model.Buses,within=pyo.Any) # Maximum value of generation of reactive power in Bus i
model.Q_gen_min = pyo.Param(model.Buses,within=pyo.Any) # Minimum value of generation of reactive power in Bus i
model.P_load_max = pyo.Param(model.Buses,within=pyo.Any) # Maximum value of load of active power in Bus i
model.P_load_min = pyo.Param(model.Buses,within=pyo.Any) # Minimum value of load of active power in Bus i
model.Q_load_max = pyo.Param(model.Buses,within=pyo.Any) # Maximum value of load of reactive power in Bus i
model.Q_load_min = pyo.Param(model.Buses,within=pyo.Any) # Minimum value of load of reactive power in Bus i

#V_slack은 차원이 없는 상수 값이므로 python에서 정의
model.V_known = pyo.Param(model.Buses,within=pyo.Any) # Known value of voltage magnitude in Bus i
model.V_setpoint = pyo.Param(model.Buses,within=pyo.Any) # Set point of voltage magnitude in Bus i
model.V_max = pyo.Param(model.Buses,within=pyo.Any) # Maximum value of voltage magnitude in Bus i
model.V_min = pyo.Param(model.Buses,within=pyo.Any) # Minimum value of voltage magnitude in Bus i

model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix

#Define variables
model.P_cal = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0) # Active power
model.Q_cal = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0) # Reactie power
model.V_cal = pyo.Var(model.Buses,within=pyo.NonNegativeReals,initialize = 1) # Voltage magnitude
model.theta_cal = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0) # Voltage angle

""" 목적함수 및 제약조건 생성"""
###Expressions
#Pmin
def P_min_rule(model, i):
    return model.alpha[i]*model.P_gen_min[i] - model.beta[i]*model.P_load_min[i] - model.omega[i]*10000000000000
model.P_min = pyo.Expression(model.Buses, rule=P_min_rule)

#Pmax
def P_max_rule(model, i):
    return model.alpha[i]*model.P_gen_max[i] - model.beta[i]*model.P_load_max[i] + model.omega[i]*10000000000000
model.P_max = pyo.Expression(model.Buses, rule=P_max_rule)

#Qmin
def Q_min_rule(model, i):
    return model.alpha[i]*model.Q_gen_min[i] - model.beta[i]*model.Q_load_min[i] - model.omega[i]*10000000000000
model.Q_min = pyo.Expression(model.Buses, rule=Q_min_rule)

#Qmax
def Q_max_rule(model, i):
    return model.alpha[i]*model.Q_gen_max[i] - model.beta[i]*model.Q_load_max[i] + model.omega[i]*10000000000000
model.Q_max = pyo.Expression(model.Buses, rule=Q_max_rule)

#P_known
def P_known_rule(model, i):
    return model.P_known[i] == model.alpha[i]*model.P_gen[i] - model.beta[i]*model.P_load[i]
model.P_known_con = pyo.Expression(model.Buses, rule=P_known_rule)

#Q_known
def Q_known_rule(model, i):
    return model.Q_known[i] == model.alpha[i]*model.Q_gen[i] - model.beta[i]*model.Q_load[i]
model.Q_known_con = pyo.Expression(model.Buses, rule=Q_known_rule)

#V_known
def V_known_rule(model, i):
    return model.V_known[i] == model.alpha[i]*model.V_setpoint[i] + model.beta[i]*1 + model.omega[i]*V_slack
model.V_known_con = pyo.Expression(model.Buses, rule=V_known_rule)

###Constraints
#Pcal
def P_cal_rule(model, i):
    return model.P_cal[i] == sum(model.V_cal[i]*model.V_cal[j]*(model.Bus_G[i,j]*math.cos(model.theta_cal[i]-model.theta_cal[j]) + model.Bus_B[i,j]*math.sin(model.theta_cal[i]-model.theta_cal[j]) ) for j in model.Buses)
model.P_cal_con = pyo.Constraint(model.Buses, rule=P_cal_rule)

#Pcal
def Q_cal_rule(model, i):
    return model.Q_cal[i] == sum(model.V_cal[i]*model.V_cal[j]*(model.Bus_G[i,j]*math.sin(model.theta_cal[i]-model.theta_cal[j]) - model.Bus_B[i,j]*math.sin(model.theta_cal[i]-model.theta_cal[j]) ) for j in model.Buses)
model.Q_cal_con = pyo.Constraint(model.Buses, rule=Q_cal_rule)

#Pmin_con
def P_min_con_rule(model, i):
    return model.P_cal[i] >= model.P_min[i]
model.P_min_con = pyo.Constraint(model.Buses, rule=P_min_con_rule)

#Pmax_con
def P_max_con_rule(model, i):
    return model.P_cal[i] <= model.P_max[i]
model.P_max_con = pyo.Constraint(model.Buses, rule=P_max_con_rule)

#Qmin_con
def Q_min_con_rule(model, i):
    return model.Q_cal[i] >= model.Q_min[i]
model.Q_min_con = pyo.Constraint(model.Buses, rule=Q_min_con_rule)

#Pmax_con
def Q_max_con_rule(model, i):
    return model.Q_cal[i] <= model.Q_max[i]
model.Q_max_con = pyo.Constraint(model.Buses, rule=Q_max_con_rule)

#V_Slack_con
def V_slack_rule(model, i):
    return model.omega[i]*model.V_cal[i] <= V_slack
model.V_slack_con = pyo.Constraint(model.Buses, rule=V_slack_rule)

#V_Slack_theta_con
def theta_slack_rule(model, i):
    return model.omega[i]*model.theta_cal[i] == 0
model.theta_slack_con = pyo.Constraint(model.Buses, rule=theta_slack_rule)

#V_known_con
def V_known_con_rule(model, i):
    return model.alpha[i]*model.V_cal[i] <= model.V_known[i]
model.V_known_con2 = pyo.Constraint(model.Buses, rule=V_known_con_rule)

### Objective function
def objective_rule(model):
    return sum ((model.alpha[i]+model.beta[i])*(model.P_known[i]-model.P_cal[i]) + model.beta[i]*(model.Q_known[i]-model.Q_cal[i])+model.alpha[i]*(model.V_known[i]-model.V_cal[i])+model.omega[i]*(V_slack-model.V_cal[i])  for i in model.Buses)
model.obj = pyo.Objective(rule=objective_rule)


"""
D. 결과 출력
"""

