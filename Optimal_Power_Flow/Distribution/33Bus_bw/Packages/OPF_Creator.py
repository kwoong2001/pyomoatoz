"""
OPF model creator
V2(예정): PU 단위에서 OPF를 풀 수 있도록 구성

V1: 모선 전압의 제곱 합 최소화 최적화 문제
   - 최적해는 모선전압의 0.95배에서 결정됨
"""

def OPF_model_creator(pyo,Bus_info):

    model = pyo.AbstractModel()

    model.Buses = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix

    # Initialization rules
    def init_Vre_rule(model, i):
        return float(Bus_info.loc[i,'vn_kv'])
    def init_Vim_rule(model, i):
        return float(0* Bus_info.loc[i,'vn_kv'])
    # Variables definition
    model.Vre = pyo.Var(model.Buses,initialize=init_Vre_rule)  # Real and imaginary phase voltages
    model.Vim = pyo.Var(model.Buses,initialize=init_Vim_rule) 

    def V_mag_2_rule(model, i):
        return model.Vre[i]**2+model.Vim[i]**2
    model.V_mag_2 = pyo.Expression(model.Buses, rule=V_mag_2_rule)

    def V_limits_rule(model, i):
        V_nom = Bus_info.loc[i,'vn_kv']
        return ((V_nom*0.95)**2,model.V_mag_2[i],(V_nom*1.05)**2)
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)

    def Objective_rule(model):
        return  sum(model.V_mag_2[i] for i in model.Buses)
    model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model