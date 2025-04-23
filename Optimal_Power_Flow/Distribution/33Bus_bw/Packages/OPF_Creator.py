"""
OPF model creator
V2(예정): PU 단위에서 OPF를 풀 수 있도록 구성
 - 정식화 부분을 다시 살펴보기

250422_V1: 모선 전압의 제곱 합 최소화 최적화 문제
   - 최적해는 모선전압의 0.95배에서 결정됨
"""

def OPF_model_creator(pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info):

    model = pyo.AbstractModel() #dat 파일을 무조건 사용해야 함

    # Load set and parameters from 'Model_data.dat' in Pre_cal_data Folder
    model.Buses = pyo.Set(dimen=1)
    model.Lines = pyo.Set(dimen=1)
    model.Loads = pyo.Set(dimen=1)
    model.Gens = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix
    model.Lines_i = pyo.Param(model.Lines,within = pyo.Any)     #Lines sending buses
    model.Lines_j = pyo.Param(model.Lines,within = pyo.Any)     #Lines receiving buses

    """
    Voltage
    """
    # Voltagae initialization rules
    # Voltage real and imaginary part definition
    model.V_re = pyo.Var(model.Buses,initialize = 1)  # Voltage magnitude
    model.V_im = pyo.Var(model.Buses,initialize = 0)  # Voltage angle

    def V_mag_2_rule(model, i):
        return model.V_re[i]**2 + model.V_im[i]**2
    model.V_mag_2 = pyo.Expression(model.Buses, rule=V_mag_2_rule)
        
    # Voltage limit
    def V_limits_rule(model, i):
        #if i ==Slackbus:
        #    return pyo.Constraint.Skip
        #else:
        return (Bus_info['min_vm_pu'][i]**2,model.V_mag_2[i],Bus_info['max_vm_pu'][i]**2)
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)
    
    """
    Power
    """
    #Generator
    model.PGen = pyo.Var(model.Buses, within=pyo.NonNegativeReals, initialize=0.0)
    model.QGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    
    #Demand
    def P_demand_rule(model,i):
        return sum(Load_info.loc[d,'p_mw']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, rule = P_demand_rule)
    def Q_demand_rule(model,i):
        return sum(Load_info.loc[d,'q_mvar']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses, rule = Q_demand_rule)

    
    """
    Balance (Generation - Demand)
    """
    # Power injection
    def Psp_rule(model, i):
        return model.PGen[i] - model.PDem[i] == 0
    model.Psp = pyo.Constraint(model.Buses,rule=Psp_rule)
    
    def Qsp_rule(model, i):
        return model.QGen[i] - model.QDem[i] == 0
    model.Qsp = pyo.Constraint(model.Buses,rule=Qsp_rule)
    
    
    def Objective_rule(model):
        return  sum(model.V_mag_2[i] for i in model.Buses)
        #return model.P_losses_total
        #return  sum(model.P_gen_cost[i] for i in model.Buses)
    model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model