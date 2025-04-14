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


""""
C. 최적화 수행
"""

""" Pyomo INPUTS """
solver = 'ipopt' # Select solver

""" Pyomo Config """
# Create the abstract model - 시그마로 표현할 수 있는 모델
model = pyo.AbstractModel()

# Loaded from csv in ../Data/
model.Buses = pyo.Set(dimen=1)

""" 목적함수 및 제약조건 생성"""

"""
D. 결과 출력
"""

