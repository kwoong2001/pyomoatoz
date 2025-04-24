"""
OPF model creator
250424_V2: PU 단위에서 OPF를 풀 수 있도록 구성

250422_V1: 모선 전압의 제곱 합 최소화 최적화 문제
   - 최적해는 모선전압의 0.95배에서 결정됨
   
제약조건 확인하는 방법: instance.(제약조건 변수 이름).display()
예시: instance.P_gen_limit_con.display()
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
    
    """
    Voltage
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    # Voltage limit
    def V_limits_rule(model, i):
        return (Bus_info['min_vm_pu'][i],model.V_mag[i],Bus_info['max_vm_pu'][i])
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)
    
    # Slack constraint
    def Slack_con_rule(model, i):
        if i == Slackbus:
            return model.V_ang[i] == 0
        else:
            return pyo.Constraint.Skip
    model.Slack_con = pyo.Constraint(model.Buses, rule=Slack_con_rule)
    
    """
    Power
    """
    #Generation variable
    model.PGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    model.QGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    
    #Active power at each node
    def P_gen_limit_rule(model,i):
        return ( (sum(Gen_info.loc[n,'min_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) , model.PGen[i], (sum(Gen_info.loc[n,'max_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) )
    model.P_gen_limit_con = pyo.Constraint(model.Buses, rule =P_gen_limit_rule)
    
    #Reactive power at each node
    def Q_gen_limit_rule(model,i):
        return ( (sum(Gen_info.loc[n,'min_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) , model.QGen[i], (sum(Gen_info.loc[n,'max_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) )
    model.Q_gen_limit_con = pyo.Constraint(model.Buses, rule =Q_gen_limit_rule)
    
    #Demand at each node
    def P_demand_rule(model,i):
        return sum(Load_info.loc[d,'p_mw']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, rule = P_demand_rule)
    def Q_demand_rule(model,i):
        return sum(Load_info.loc[d,'q_mvar']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses, rule = Q_demand_rule)

    #Active power flow in each line
    if 1==0:
        def P_line_flow_sending_rule(model,l):
            i = Line_info.loc[l,'from_bus']
            j = Line_info.loc[l,'to_bus']
            return model.V_mag[i] * model.V_mag[i] * model.Bus_G[i,i] + model.V_mag[i] * model.V_mag[j] * ( model.Bus_G[i,j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) + model.Bus_B[i,j]* pyo.sin(model.V_ang[i]-model.V_ang[j]) )
        model.P_line_flow_sending = pyo.Expression(model.Lines,rule = P_line_flow_sending_rule)
    
    """
    Balance (Generation - Demand)
    """
    # Power injection at each node
    def P_bal_rule(model, i):
        return model.PGen[i] - model.PDem[i] == model.V_mag[i] * ( sum (model.V_mag[j] * ( model.Bus_G[i,j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) + model.Bus_B[i,j]* pyo.sin(model.V_ang[i]-model.V_ang[j]) ) for j in model.Buses ) )
    model.P_bal_con = pyo.Constraint(model.Buses,rule=P_bal_rule)
    
    def Q_bal_rule(model, i):
        return model.QGen[i] - model.QDem[i] == model.V_mag[i] * ( sum (model.V_mag[j] * ( model.Bus_G[i,j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j]* pyo.cos(model.V_ang[i]-model.V_ang[j]) ) for j in model.Buses ) )
    model.Q_bal_con = pyo.Constraint(model.Buses,rule=Q_bal_rule)
    
    # Power injection value (Gen - Demand)
    def P_inj_rule(model,i):
        return model.PGen[i] - model.PDem[i]
    model.P_inj = pyo.Expression(model.Buses,rule = P_inj_rule)
    def Q_inj_rule(model,i):
        return model.QGen[i] - model.QDem[i]
    model.Q_inj = pyo.Expression(model.Buses,rule = Q_inj_rule)
    
    
    def Objective_rule(model):
        return  sum(model.PGen[i] for i in model.Buses)
        #return model.P_losses_total
        #return  sum(model.P_gen_cost[i] for i in model.Buses)
    model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model