"""
OPF model creator for balanced system
- PU를 적용하여 OPF를 풀 수 있는 시스템에 적용 가능
- Unbalanced system은 PU를 적용하기에 까다로울 것임

250709_V10: 무효전력 출력 범위 및 식 수정

250626_V9: Multi-period 수행이 가능한 버젼으로 만들기 위한 초기 작업

250618_V8: Generator 관련 제약조건 및 Cost 반영 정보 수정

250611_V7: Line loss 오류 수정

250507_V6: Matpower 용으로 변환

250501_V5: 선로의 상태를 고려한 최적화 문제를 구현하는 OPF_model_creator_with_switch 함수 생성 - G와 B가 Line status에 따라 재구성될 필요가 있음

250425_V4: 선로 rating 제약조건 반영, Sending and receiving power and current 반영

250424_V3: 발전 비용을 반영한 OPF 문제 구성

250424_V2: PU 단위에서 OPF를 풀 수 있도록 구성

250422_V1: 모선 전압의 제곱 합 최소화 최적화 문제
   - 최적해는 모선전압의 0.95배에서 결정됨
   
제약조건 확인하는 방법: instance.(제약조건 변수 이름).display()
예시: instance.P_gen_limit_con.display()

변수 해 확인하는 방법: instance.(변수 이름)[인덱스].value
예시: instance.V_mag[bus].value

Expression 값 확인하는 방법: instance.(expression 변수 이름)[인덱스].value
예시: instance.Q_line_loss[line].expr()
"""

# Line 상태를 반영하지 않는 일반 OPF 함수
def OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info):

    model = pyo.AbstractModel() #dat 파일을 무조건 사용해야 함

    """
    Set and parameters
    
    """
    # Load set and parameters from 'Model_data.dat' in Pre_cal_data Folder
    model.Buses = pyo.Set(dimen=1)
    model.Lines = pyo.Set(dimen=1)
    model.Loads = pyo.Set(dimen=1)
    model.Gens = pyo.Set(dimen=1)
    model.Times = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix
    
    #Demand at each node in time t - Unit:PU
    def P_demand_rule(model,i,t):
        return sum(Load_info.loc[d,'p_mw_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, model.Times, rule = P_demand_rule)
    def Q_demand_rule(model,i,t):
        return sum(Load_info.loc[d,'q_mvar_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses,model.Times, rule = Q_demand_rule)
    
    """
    Variables
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,model.Times,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,model.Times,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    #Generation variable
    model.PGen = pyo.Var(model.Gens, model.Buses, model.Times, within=pyo.NonNegativeReals, initialize=0.0)
    model.QGen = pyo.Var(model.Gens, model.Buses, model.Times, within=pyo.Reals, initialize=0.0)
    
    """
    Expressions - Flow - Equation (2) - (4), (7)
    """
    # Equation (3)
    def P_line_flow_sending_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i,t] * model.V_mag[i,t] + model.Bus_G[i,j] * model.V_mag[i,t]* model.V_mag[j,t] * pyo.cos(model.V_ang[i,t]-model.V_ang[j,t]) + model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[j,t] * pyo.sin(model.V_ang[i,t]-model.V_ang[j,t]))
    model.P_line_flow_sending = pyo.Expression(model.Lines,model.Times,rule = P_line_flow_sending_rule)
    
    # Equation (7)
    def Q_line_flow_sending_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[i,t] + model.Bus_G[i,j] * model.V_mag[i,t]* model.V_mag[j,t] * pyo.sin(model.V_ang[i,t]-model.V_ang[j,t]) - model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[j,t] * pyo.cos(model.V_ang[i,t]-model.V_ang[j,t]))
    model.Q_line_flow_sending = pyo.Expression(model.Lines,model.Times,rule = Q_line_flow_sending_rule)
    
    # Equation (4)
    def P_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return ((-1) * model.Bus_G[i, j] * model.V_mag[j, t] * model.V_mag[j, t]
                + model.Bus_G[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t])
                - model.Bus_B[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.sin(model.V_ang[i, t] - model.V_ang[j, t]))
    model.P_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=P_line_flow_receiving_rule)

    def Q_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Bus_B[i, j] * model.V_mag[j, t] * model.V_mag[j, t]
                - model.Bus_G[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.sin(model.V_ang[i, t] - model.V_ang[j, t])
                - model.Bus_B[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t]))
    model.Q_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=Q_line_flow_receiving_rule)
    
    # Equation (2)
    def P_line_loss_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.P_line_flow_sending[l,t] + model.P_line_flow_receiving[l,t]) * base_MVA
    model.P_line_loss = pyo.Expression(model.Lines, model.Times, rule = P_line_loss_rule)
    
    def Q_line_loss_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Q_line_flow_sending[l, t] + model.Q_line_flow_receiving[l, t]) * base_MVA
    model.Q_line_loss = pyo.Expression(model.Lines, model.Times, rule=Q_line_loss_rule)

    def S_line_flow_sending_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return model.P_line_flow_sending[l, t] ** 2 + model.Q_line_flow_sending[l, t] ** 2
    model.S_line_flow_sending = pyo.Expression(model.Lines, model.Times, rule=S_line_flow_sending_rule)

    def S_line_flow_sending_con_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']

        if Line_info.loc[l, 'rate_MVA'] == 0:
            return pyo.Constraint.Skip
        else:
            return model.S_line_flow_sending[l, t] <= (Line_info.loc[l, 'rate_MVA'] / base_MVA) ** 2
    model.S_line_flow_sending_con = pyo.Constraint(model.Lines, model.Times, rule=S_line_flow_sending_con_rule)

    def S_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return model.P_line_flow_receiving[l, t] ** 2 + model.Q_line_flow_receiving[l, t] ** 2
    model.S_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=S_line_flow_receiving_rule)

    def S_line_flow_receiving_con_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']

        if Line_info.loc[l, 'rate_MVA'] == 0:
            return pyo.Constraint.Skip
        else:
            return model.S_line_flow_receiving[l, t] <= (Line_info.loc[l, 'rate_MVA'] / base_MVA) ** 2
    model.S_line_flow_receiving_con = pyo.Constraint(model.Lines, model.Times, rule=S_line_flow_receiving_con_rule)
    
    """
    Constraints - Load Balance (Generation - Demand) - Equation (5)-(6)
    2 Constraints
    """
    # Power injection at each node
    def P_bal_rule(model, i, t):
        return (
            sum(model.PGen[n, i, t] for n in model.Gens)
            - model.PDem[i, t]
            == (
                sum(
                    model.P_line_flow_sending[l, t]
                    for l in Line_info.index
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum(
                    model.P_line_flow_receiving[l, t]
                    for l in Line_info.index
                    if Line_info.loc[l, "to_bus"] == i
                )
            )
        )
    model.P_bal_con = pyo.Constraint(model.Buses, model.Times, rule=P_bal_rule)

    def Q_bal_rule(model, i, t):
        return (
            sum(model.QGen[n, i, t] for n in model.Gens)
            - model.QDem[i, t]
            == (
                sum(
                    model.Q_line_flow_sending[l, t]
                    for l in Line_info.index
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum(
                    model.Q_line_flow_receiving[l, t]
                    for l in Line_info.index
                    if Line_info.loc[l, "to_bus"] == i
                )
                + (-1)*sum(model.Bus_B[i,m] for m in model.Buses)*model.V_mag[i,t] * model.V_mag[i,t]
            )
        )
    model.Q_bal_con = pyo.Constraint(model.Buses, model.Times, rule=Q_bal_rule)
    
    """
    Constraints - Power and voltage - Equation (8)-(11)
    - Power
    4 Constraints
    2 Expressions - Convert PU to MW,MVar
    
    - Voltage
    2 Constraints
    2 Expression - Voltage magnitude [kV], angle [deg]
    """
    
    # Equation (8) - Min, Unit: [PU]
    # Equation (8) - Min, Unit: [PU]
    def P_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            return Gen_info.loc[n, 'min_p_mw'] / base_MVA <= model.PGen[n, i, t]
        else:
            return model.PGen[n, i, t] <= 0
    model.P_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_min_rule)

    # Equation (8) - Max, Unit: [PU]
    def P_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            return model.PGen[n, i, t] <= Gen_info.loc[n, 'max_p_mw'] / base_MVA
        else:
            return model.PGen[n, i, t] <= 0
    model.P_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_max_rule)

    # Equation (9) - Min, Unit: [PU]
    def Q_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            return Gen_info.loc[n, 'min_q_mvar'] / base_MVA <= model.QGen[n, i, t]
        else:
            return model.QGen[n, i, t] <= 0
    model.Q_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_min_rule)

    # Equation (9) - Max, Unit: [PU]
    def Q_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            return model.QGen[n, i, t] <= Gen_info.loc[n, 'max_q_mvar'] / base_MVA
        else:
            return model.QGen[n, i, t] <= 0
    model.Q_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_max_rule)

    # Expression - Active power expression at each node - Unit:MW
    def P_gen_MW_rule(model, n, i, t):
        return model.PGen[n, i, t] * base_MVA
    model.P_gen_MW = pyo.Expression(model.Gens, model.Buses, model.Times, rule=P_gen_MW_rule)

    # Expression - Reactive power expression at each node - Unit:MVar
    def Q_gen_MVar_rule(model, n, i, t):
        return model.QGen[n, i, t] * base_MVA
    model.Q_gen_MVar = pyo.Expression(model.Gens, model.Buses, model.Times, rule=Q_gen_MVar_rule)
    
    # Equation (10), Unit: [PU]
    def V_limits_rule(model, i,t):
        return (Bus_info['Vmin_pu'][i],model.V_mag[i,t],Bus_info['Vmax_pu'][i])
    model.V_limits_con = pyo.Constraint(model.Buses, model.Times, rule=V_limits_rule)
    
    # Equation (11)
    def Slack_con_rule(model, i, t):
        if i == Slackbus:
            return model.V_ang[i,t] == 0
        else:
            return pyo.Constraint.Skip
    model.Slack_con = pyo.Constraint(model.Buses, model.Times, rule=Slack_con_rule)
    
    # Expression - Voltage expression - magnitude [kV]
    def V_mag_kv_rule(model, i,t):
        return model.V_mag[i,t] * Bus_info.loc[i,'baseKV']
    model.V_mag_kv = pyo.Expression(model.Buses, model.Times, rule=V_mag_kv_rule)
    # Expression - Voltage expression - angle [deg]
    def V_ang_deg_rule(model, i,t):
        return model.V_ang[i,t] * 180 / np.pi
    model.V_ang_deg = pyo.Expression(model.Buses, model.Times, rule=V_ang_deg_rule)
    
    """
    Constraints - Currents - Equation (12)-(14)
    1 Constraint
    3 Expressions
    """
    # Equation (13)
    def I_line_re_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (
            (-1) * model.Bus_G[i, j] * model.V_mag[i, t] * pyo.cos(model.V_ang[i, t])
            + model.Bus_B[i, j] * model.V_mag[i, t] * pyo.sin(model.V_ang[i, t])
            + model.Bus_G[i, j] * model.V_mag[j, t] * pyo.cos(model.V_ang[j, t])
            - model.Bus_B[i, j] * model.V_mag[j, t] * pyo.sin(model.V_ang[j, t])
        )
    model.I_line_re = pyo.Expression(model.Lines, model.Times, rule=I_line_re_rule)

    # Equation (14)
    def I_line_im_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (
            (-1) * model.Bus_B[i, j] * model.V_mag[i, t] * pyo.cos(model.V_ang[i, t])
            - model.Bus_G[i, j] * model.V_mag[i, t] * pyo.sin(model.V_ang[i, t])
            + model.Bus_B[i, j] * model.V_mag[j, t] * pyo.cos(model.V_ang[j, t])
            + model.Bus_G[i, j] * model.V_mag[j, t] * pyo.sin(model.V_ang[j, t])
        )
    model.I_line_im = pyo.Expression(model.Lines, model.Times, rule=I_line_im_rule)
    
    # Equation (12) - Sum of squares
    def I_line_sq_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.I_line_re[l,t] ** 2 + model.I_line_im[l,t] ** 2)
    model.I_line_sq = pyo.Expression(model.Lines, model.Times ,rule = I_line_sq_rule)
    
    # Equation (12) - 수렴에 영향을 줌, S_line_flow_limit으로 대체
    if 1==0:
        def I_loading_con_rule(model,l,t):
            base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
            return model.I_line_sq[l,t] <= (9999999/base_current) ** 2                # If line limit info is existed, the limit replaces 999999.
        model.I_loading_con = pyo.Constraint(model.Lines, model.Times, rule = I_loading_con_rule)
    
    """
    Minimize Generation cost
    1 Expressions
    """
    if 1 == 0: 
        def P_cost_rule(model, n,i):
            if Gen_info.loc[n,'bus'] == i:
                try: # 2차항까지 있는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i]*base_MVA + Gen_info.loc[n,'c2'] * (model.PGen[n,i]*base_MVA)**2
                except: # 2차항이 없는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i]*base_MVA
            else:
                return 0.0
        model.P_cost = pyo.Expression(model.Gens, model.Buses,rule=P_cost_rule)
        
        def Objective_rule(model):
            return sum(sum(model.P_cost[n,i] for n in model.Gens) for i in model.Buses)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    """
    Objective Function - Equation (1)
     - Minimize loss
    """
    if 1 == 1: 
        # Equation (1)
        def Objective_rule(model):
            return sum(model.P_line_loss[l,t] for l in model.Lines for t in model.Times)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model
   

# Line 상태를 반영할 수 있는 OPF 함수
def OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info):

    model = pyo.AbstractModel() #dat 파일을 무조건 사용해야 함

    """
    Set and parameters
    
    """
    # Load set and parameters from 'Model_data.dat' in Pre_cal_data Folder
    model.Buses = pyo.Set(dimen=1)
    model.Lines = pyo.Set(dimen=1)
    model.Loads = pyo.Set(dimen=1)
    model.Gens = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix
    
    #Demand at each node - Unit:PU
    def P_demand_rule(model,i):
        return sum(Load_info.loc[d,'p_mw']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, rule = P_demand_rule)
    def Q_demand_rule(model,i):
        return sum(Load_info.loc[d,'q_mvar']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses, rule = Q_demand_rule)
    
    """
    Variables
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    #Generation variable
     # Generation Initialization
    def P_gen_ini_rule(model,n,i):
        if Gen_info.loc[n,'bus'] == i:
            return Gen_info.loc[n,'p_mw']/base_MVA
        else:
            return 0.0
    
    model.PGen = pyo.Var(model.Gens, model.Buses, within=pyo.NonNegativeReals, initialize=P_gen_ini_rule)
    model.QGen = pyo.Var(model.Gens, model.Buses, within=pyo.Reals, initialize=0.0)
    
    """
    Expressions - Flow - Equation (2) - (4), (7)
    """
    # Equation (3)
    def P_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) + model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]))
    model.P_line_flow_sending = pyo.Expression(model.Lines,rule = P_line_flow_sending_rule)
    
    # Equation (7)
    def Q_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]))
    model.Q_line_flow_sending = pyo.Expression(model.Lines,rule = Q_line_flow_sending_rule)
    
    # Equation (4)
    def P_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[j] * model.V_mag[j] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]))
    model.P_line_flow_receiving = pyo.Expression(model.Lines,rule = P_line_flow_receiving_rule)    
    
    def Q_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[j] * model.V_mag[j] - model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]))
    model.Q_line_flow_receiving = pyo.Expression(model.Lines,rule = Q_line_flow_receiving_rule)
    
    # Equation (2)
    def P_line_loss_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.P_line_flow_sending[l] + model.P_line_flow_receiving[l]) * base_MVA
    model.P_line_loss = pyo.Expression(model.Lines,rule = P_line_loss_rule)
    
    def Q_line_loss_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Q_line_flow_sending[l] + model.Q_line_flow_receiving[l]) * base_MVA
    model.Q_line_loss = pyo.Expression(model.Lines,rule = Q_line_loss_rule)
    
    def S_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return model.P_line_flow_sending[l]** 2 + model.Q_line_flow_sending[l]** 2
    model.S_line_flow_sending = pyo.Expression(model.Lines,rule = S_line_flow_sending_rule)
    
    def S_line_flow_sending_con_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        
        if Line_info.loc[l,'rate_MVA'] == 0:
            return pyo.Constraint.Skip
        else:
            return model.S_line_flow_sending[l] <= (Line_info.loc[l,'rate_MVA'] / base_MVA) ** 2
    model.S_line_flow_sending_con = pyo.Constraint(model.Lines,rule = S_line_flow_sending_con_rule)
    
    def S_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return model.P_line_flow_receiving[l]** 2 + model.Q_line_flow_receiving[l]** 2
    model.S_line_flow_receiving = pyo.Expression(model.Lines,rule = S_line_flow_receiving_rule)
    
    def S_line_flow_receiving_con_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        
        if Line_info.loc[l,'rate_MVA'] == 0:
            return pyo.Constraint.Skip
        else:
            return model.S_line_flow_receiving[l] <= (Line_info.loc[l,'rate_MVA'] / base_MVA) ** 2
    model.S_line_flow_receiving_con = pyo.Constraint(model.Lines,rule = S_line_flow_receiving_con_rule)
    
    """
    Line status
    1 Variables
    1 Constraints
    Expressions
    Recreation of G and B
    """
    ##Line status variable
    model.Line_Status = pyo.Var(model.Lines, within=pyo.Integers, bounds=(0, 1),initialize = 0)
    
    ##Line status constraint
    def Line_status_rule(model,l):
        #line_con = Line_info[['from_bus','to_bus']].values.tolist()
        
        return model.Line_Status[l] <= 1
                        
    model.Line_Status_con = pyo.Constraint(model.Lines, rule = Line_status_rule)
    
    
    # Power injection at each node
    def P_bal_rule(model, i):
        return sum(model.PGen[n,i] for n in model.Gens) - model.PDem[i] == ( sum (model.Line_Status[l]*model.P_line_flow_sending[l] for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.Line_Status[l]*model.P_line_flow_receiving[l] for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) )
    model.P_bal_con = pyo.Constraint(model.Buses,rule=P_bal_rule)
    
    def Q_bal_rule(model, i):
        return sum(model.QGen[n,i] for n in model.Gens) - model.QDem[i] == ( sum (model.Line_Status[l]*model.Q_line_flow_sending[l] for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.Line_Status[l]*model.Q_line_flow_receiving[l] for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) ) + (-1)*sum(model.Bus_B[i,m] for m in model.Buses)*model.V_mag[i] * model.V_mag[i]
    model.Q_bal_con = pyo.Constraint(model.Buses,rule=Q_bal_rule)
    
    
    """
    Constraints - Power and voltage - Equation (8)-(11)
    - Power
    4 Constraints
    2 Expressions - Convert PU to MW,MVar
    
    - Voltage
    2 Constraints
    2 Expression - Voltage magnitude [kV], angle [deg]
    """
    
    # Equation (8) - Min, Unit: [PU]
    def P_gen_min_rule(model,n,i):
        if Gen_info.loc[n,'bus'] == i:
            return Gen_info.loc[n,'min_p_mw']/base_MVA <= model.PGen[n,i]
        else:
            return model.PGen[n,i] <= 0
    model.P_gen_min_con = pyo.Constraint(model.Gens, model.Buses, rule =P_gen_min_rule)
    
    # Equation (8) - Max, Unit: [PU]
    def P_gen_max_rule(model,n,i):
        if Gen_info.loc[n,'bus'] == i:
            return model.PGen[n,i] <= Gen_info.loc[n,'max_p_mw']/base_MVA
        else:
            return model.PGen[n,i] <= 0
    model.P_gen_max_con = pyo.Constraint(model.Gens, model.Buses, rule =P_gen_max_rule)
    
    # Equation (9) - Min, Unit: [PU]
    def Q_gen_min_rule(model,n,i):
        if Gen_info.loc[n,'bus'] == i:
            return Gen_info.loc[n,'min_q_mvar']/base_MVA <= model.QGen[n,i]
        else:
            return model.QGen[n,i] <= 0
    model.Q_gen_min_con = pyo.Constraint(model.Gens, model.Buses, rule =Q_gen_min_rule)
    
    # Equation (9) - Max, Unit: [PU]
    def Q_gen_max_rule(model,n,i):
        if Gen_info.loc[n,'bus'] == i:
            return model.QGen[n,i] <= Gen_info.loc[n,'max_q_mvar']/base_MVA
        else:
            return model.QGen[n,i] <= 0
    model.Q_gen_max_con = pyo.Constraint(model.Gens, model.Buses, rule =Q_gen_max_rule)
    
    # Expression - Active power expression at each node - Unit:MW
    def P_gen_MW_rule(model,n,i):
        return  model.PGen[n,i] * base_MVA
    model.P_gen_MW = pyo.Expression(model.Gens, model.Buses, rule =P_gen_MW_rule)
    
    # Expression - Reactive power expression at each node - Unit:MW
    def Q_gen_MVar_rule(model,n,i):
        return  model.QGen[n,i] * base_MVA
    model.Q_gen_MVar = pyo.Expression(model.Gens, model.Buses, rule =Q_gen_MVar_rule)
    
    # Equation (10), Unit: [PU]
    def V_limits_rule(model, i):
        return (Bus_info['Vmin_pu'][i],model.V_mag[i],Bus_info['Vmax_pu'][i])
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)
    
    # Equation (11)
    def Slack_con_rule(model, i):
        if i == Slackbus:
            return model.V_ang[i] == 0
        else:
            return pyo.Constraint.Skip
    model.Slack_con = pyo.Constraint(model.Buses, rule=Slack_con_rule)
    
    # Expression - Voltage expression - magnitude [kV]
    def V_mag_kv_rule(model, i):
        return model.V_mag[i] * Bus_info.loc[i,'baseKV']
    model.V_mag_kv = pyo.Expression(model.Buses, rule=V_mag_kv_rule)
    # Expression - Voltage expression - angle [deg]
    def V_ang_deg_rule(model, i):
        return model.V_ang[i] * 180 / np.pi
    model.V_ang_deg = pyo.Expression(model.Buses, rule=V_ang_deg_rule)
    
    """
    Constraints - Currents - Equation (12)-(14)
    1 Constraint
    3 Expressions
    """
    # Equation (13)
    def I_line_re_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ( (-1)*model.Bus_G[i,j]*model.V_mag[i]*pyo.cos(model.V_ang[i]) + model.Bus_B[i,j]*model.V_mag[i]*pyo.sin(model.V_ang[i]) + model.Bus_G[i,j]*model.V_mag[j]*pyo.cos(model.V_ang[j]) - model.Bus_B[i,j]*model.V_mag[j]*pyo.sin(model.V_ang[j]) )
    model.I_line_re = pyo.Expression(model.Lines,rule = I_line_re_rule)
    
    # Equation (14)
    def I_line_im_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1)*model.Bus_B[i,j]*model.V_mag[i]*pyo.cos(model.V_ang[i]) - model.Bus_G[i,j]*model.V_mag[i]*pyo.sin(model.V_ang[i]) + model.Bus_B[i,j]*model.V_mag[j]*pyo.cos(model.V_ang[j]) + model.Bus_G[i,j]*model.V_mag[j]*pyo.sin(model.V_ang[j]))
    model.I_line_im = pyo.Expression(model.Lines,rule = I_line_im_rule)
    
    # Equation (12) - Sum of squares
    def I_line_sq_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.I_line_re[l] ** 2 + model.I_line_im[l] ** 2)
    model.I_line_sq = pyo.Expression(model.Lines,rule = I_line_sq_rule)
    
    # Equation (12) - 수렴에 영향을 줌, S_line_flow_limit으로 대체
    if 1==0:
        def I_loading_con_rule(model,l):
            base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
            return model.I_line_sq[l] <= (9999999/base_current) ** 2                # If line limit info is existed, the limit replaces 999999.
        model.I_loading_con = pyo.Constraint(model.Lines,rule = I_loading_con_rule)
    
    """
    Minimize Generation cost
    1 Expressions
    """
    if 1 == 0: 
        def P_cost_rule(model, n,i):
            if Gen_info.loc[n,'bus'] == i:
                try: # 2차항까지 있는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i]*base_MVA + Gen_info.loc[n,'c2'] * (model.PGen[n,i]*base_MVA)**2
                except: # 2차항이 없는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i]*base_MVA
            else:
                return 0.0
        model.P_cost = pyo.Expression(model.Gens, model.Buses,rule=P_cost_rule)
        
        def Objective_rule(model):
            return sum(sum(model.P_cost[n,i] for n in model.Gens) for i in model.Buses)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    """
    Objective Function - Equation (1)
     - Minimize loss
    """
    if 1 == 1: 
        # Equation (1)
        def Objective_rule(model):
            return sum(model.P_line_loss[l] for l in model.Lines)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model