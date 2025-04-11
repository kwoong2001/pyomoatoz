import pyomo.environ as pyo
import numpy as np
import pandas as pd
import math

from pyomo.environ import *

def read(file_path: str, sheet_name: str):
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    return df.to_numpy()

file_path = 'C:\\project\\github\\pyomoatoz#\\nthPractice\\ac_case25.xlsx'
BUS = read(file_path, 'bus')
GEN = read(file_path, 'generator')
TR = read(file_path, 'transformer')
LINE = read(file_path, 'branch')


B = BUS.shape[0]
L0 = LINE.shape[0]
l = LINE.shape[1]
G = GEN.shape[0]
T = TR.shape[0]

mask = np.zeros((L0,1))
TRad = np.zeros((L0,l))

mask = np.zeros(L0, dtype=bool)
for i in range(T):
    mask_cal = (LINE[:, 0] == TR[i, 0]) & (LINE[:, 1] == TR[i, 1])
    mask |= mask_cal

TRad = LINE[mask, :]
LINE = LINE[~mask, :]
line = (LINE[:, 0:2].astype(int)-1)
trad = (TRad[:, 0:2].astype(int)-1)
L = LINE.shape[0]

Y = np.zeros((B,B),dtype=complex)



for i in range(0,L):  
    Y[line[i, 0], line[i, 1]] -= 1 / (LINE[i, 2] + LINE[i, 3]*1j)
    Y[line[i, 1], line[i, 0]] = Y[line[i, 0], line[i, 1]]
    
    for j in range(0,2):  
        Y[line[i, j], line[i, j]] += 1 / (LINE[i, 2] + LINE[i, 3]*1j) + (LINE[i, 4]*1j) / 2

for i in range(0,T):
    Y[trad[i, 0], trad[i, 1]] -= TR[i, 2] / (TRad[i, 2] + TRad[i, 3]*1j)
    Y[trad[i, 1], trad[i, 0]] = Y[trad[i, 0], trad[i, 1]]
    
    for j in range(0,2):
        if j == 0:
            Y[trad[i, j], trad[i, j]] += (TR[i, 2]**2) / (TRad[i, 2] + TRad[i, 3]*1j) + TRad[i, 4]*1j
        elif j == 1:
            Y[trad[i, j], trad[i, j]] += 1 / (TRad[i, 2] + TRad[i, 3]*1j) + TRad[i, 4]*1j



PL = BUS[:,2]
PL = PL.reshape(-1,1)
QL = BUS[:,3]
QL = QL.reshape(-1,1)

PG = np.zeros((B,1))
QG = np.zeros((B,1))

for i in range(G):
    PG[int(GEN[i,1]),0] = PG[int(GEN[i,1]),0] + GEN[i,2]
    QG[int(GEN[i,1]),0] = QG[int(GEN[i,1]),0] + GEN[i,3] 

P = PL - PG
Q = QL - QG

V = np.zeros((B,1),dtype=complex)
V = BUS[:,5].reshape(-1,1).astype(complex)

for i in range(G):
    V[int(GEN[i,1]),0] = GEN[i,6]


Vmag = np.abs(V)
Vang = np.angle(V)



P = P.flatten()
Q = Q.flatten()
Vmag = Vmag.flatten()
Vang = Vang.flatten()

n = len(P)
print(n)

init_P = {i: float(P[i]) for i in range(n)}
init_Q = {i: float(Q[i]) for i in range(n)}
init_Vmag = {i: float(Vmag[i]) for i in range(n)}
init_Vang = {i: float(Vang[i]) for i in range(n)}

solver = 'ipopt'

model = pyo.ConcreteModel

model.Bus = pyo.RangeSet(0, n-1)

model.P = pyo.Param(model.Bus, initialize=init_P)
model.Q = pyo.Param(model.Bus, initialize=init_Q)
model.Vmag = pyo.Param(model.Bus, initialize=init_Vmag)
model.Vang = pyo.Param(model.Bus, initialize=init_Vang)