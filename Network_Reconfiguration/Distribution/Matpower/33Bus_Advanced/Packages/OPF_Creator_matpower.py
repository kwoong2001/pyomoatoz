"""
OPF model creator for balanced system
- PU를 적용하여 OPF를 풀 수 있는 시스템에 적용 가능
- Unbalanced system은 PU를 적용하기에 까다로울 것임

250911_v20: Line status interval con 추가(시간단위별 계통 재구성)

~250903_v19: without_switch 에도 Line status 변수 추가(시각화용), Net load, PV curtailment Expression 추가

250801_V13: Line status는 시간에 관계없이 동일하도록 설정(시간별로 바꾸게 되면 계산량이 증가하여 수렴 시간이 너무 길어짐)

250730_V12: Radiality constraint 반영

250729_V11: Multi-period 반영

250709_V10: 무효전력 출력 범위 및 충전용량 반영

250618_V8: Generator 관련 제약조건 및 Cost 반영 정보 수정

250611_v7: Cost 최소화와 Loss 최소화 분리

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

Expression 값 확인하는 방법: instance.(expression 변수 이름)[인덱스].expr()
예시: instance.Q_line_loss[line].expr()
"""

# Line On/Off 수행하지 않는 OPF 함수
def OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,DG_profile_df,Load_profile_df):

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

    def Line_Status_info(model,l):
        return Line_info.loc[l,'in_service']
    model.Line_Status = pyo.Var(model.Lines, within=pyo.Binary, rule = Line_Status_info)  # Line status (1: On, 0: Off)

    #Demand at each node in time t - Unit:PU
    def P_demand_rule(model,i,t):
        return sum(Load_profile_df.loc[d,'p_mw_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, model.Times, rule = P_demand_rule)
    def Q_demand_rule(model,i,t):
        return sum(Load_profile_df.loc[d,'q_mvar_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses,model.Times, rule = Q_demand_rule)
    
    #Distributed generation at each node in time t - Unit:PU
    def P_dg_rule(model,n,i,t):
        if i == Slackbus:
            return 0
        else:
            if Gen_info.loc[n,'bus']==i:
                return DG_profile_df.loc[n,'p_mw_'+str(t)]/base_MVA 
            else:
                return 0
    model.PDg = pyo.Expression(model.Gens, model.Buses, model.Times, rule = P_dg_rule)
    def Q_dg_rule(model,n,i,t):
        if i == Slackbus:
            return 0
        else:
            if Gen_info.loc[n,'bus']==i:
                return DG_profile_df.loc[n,'q_mvar_'+str(t)]/base_MVA
            else:
                return 0
    model.QDg = pyo.Expression(model.Gens, model.Buses,model.Times, rule = Q_dg_rule)
    
    """
    Variables
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,model.Times,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,model.Times,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    #Generation variable
    model.PGen = pyo.Var(model.Gens, model.Buses, model.Times, within=pyo.Reals, initialize=0.0)
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

    def S_line_flow_sending_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return model.P_line_flow_sending[l, t] ** 2 + model.Q_line_flow_sending[l, t] ** 2
    model.S_line_flow_sending = pyo.Expression(model.Lines, model.Times, rule=S_line_flow_sending_rule)

    if 1==0:
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

    if 1==0:
        def S_line_flow_receiving_con_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']

            if Line_info.loc[l, 'rate_MVA'] == 0:
                return pyo.Constraint.Skip
            else:
                return model.S_line_flow_receiving[l, t] <= (Line_info.loc[l, 'rate_MVA'] / base_MVA) ** 2
        model.S_line_flow_receiving_con = pyo.Constraint(model.Lines, model.Times, rule=S_line_flow_receiving_con_rule)
        
    # Equation (2)
    def P_line_loss_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ( - model.Bus_G[i, j] *
                ( model.V_mag[i, t] ** 2 +  model.V_mag[j, t] ** 2
                    - 2 * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t]))
                ) * base_MVA
    model.P_line_loss = pyo.Expression(model.Lines, model.Times, rule = P_line_loss_rule)
    
    def Q_line_loss_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Q_line_flow_sending[l, t] + model.Q_line_flow_receiving[l, t]) * base_MVA
    model.Q_line_loss = pyo.Expression(model.Lines, model.Times, rule=Q_line_loss_rule)
    
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
                    for l in model.Lines
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum( 
                    model.P_line_flow_receiving[l, t]
                    for l in model.Lines
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
                    for l in model.Lines
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum(
                    model.Q_line_flow_receiving[l, t]
                    for l in model.Lines
                    if Line_info.loc[l, "to_bus"] == i
                )
                - sum(Line_info.loc[l,'b_pu']/2  for l in model.Lines if Line_info.loc[l, "from_bus"] == i) * model.V_mag[i,t] * model.V_mag[i,t]
                - sum(Line_info.loc[l,'b_pu']/2  for l in model.Lines if Line_info.loc[l, "to_bus"] == i) * model.V_mag[i,t] * model.V_mag[i,t]
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
    def P_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return Gen_info.loc[n, 'min_p_mw'] / base_MVA <= model.PGen[n, i, t]
            else:
                return model.PDg[n,i,t] <= model.PGen[n, i, t]
        else:
            return 0 <= model.PGen[n, i, t]
    model.P_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_min_rule)

    # Equation (8) - Max, Unit: [PU]
    def P_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return model.PGen[n, i, t] <= Gen_info.loc[n, 'max_p_mw'] / base_MVA
            else:
                return model.PGen[n, i, t] <= model.PDg[n,i,t]
        else:
            return model.PGen[n, i, t] <= 0
    model.P_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_max_rule)

    # Equation (9) - Min, Unit: [PU]
    def Q_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return Gen_info.loc[n, 'min_q_mvar'] / base_MVA <= model.QGen[n, i, t]
            else:
                return model.QDg[n,i,t] * (-1) <= model.QGen[n, i, t]
                #return 0 <= model.QGen[n, i, t]
        else:
            return 0 <= model.QGen[n, i, t]
    model.Q_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_min_rule)

    # Equation (9) - Max, Unit: [PU]
    def Q_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return model.QGen[n, i, t] <= Gen_info.loc[n, 'max_q_mvar'] / base_MVA
            else:
                return model.QGen[n, i, t] <= model.QDg[n,i,t]
                #return model.QGen[n, i, t] <= 0
        else:
            return model.QGen[n, i, t] <= 0
    model.Q_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_max_rule)

    """
    Expressions - Curtailment
    """

    # Expression - PV curtailment - Unit:MW
    def PV_curtailment_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return 0
            else:
                return (model.PDg[n, i, t] - model.PGen[n, i, t]) * base_MVA
        else:
            return 0
    model.PV_curtailment_value = pyo.Expression(model.Gens, model.Buses, model.Times, rule=PV_curtailment_rule)

    """
    Expression - Net load
    """

    def Net_load_rule(model, i, t):
        return (model.PDem[i, t] - sum(model.PGen[n, i, t] for n in model.Gens if Gen_info.loc[n, 'bus'] == i)) * base_MVA
    
    model.Net_load = pyo.Expression(model.Buses, model.Times, rule=Net_load_rule)


    
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
            return ((-1)*np.pi,model.V_ang[i,t],np.pi)
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
    if 1==0:
        # Equation (13)
        def I_line_re_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']
            return(
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
            return(
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
    if 1 == 1: 
        def P_cost_rule(model, n,i,t):
            if Gen_info.loc[n,'bus'] == i:
                try: # 2차항까지 있는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i,t]*base_MVA + Gen_info.loc[n,'c2'] * (model.PGen[n,i,t]*base_MVA)**2
                except: # 2차항이 없는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i,t]*base_MVA
            else:
                return 0.0
        model.P_cost = pyo.Expression(model.Gens, model.Buses, model.Times,rule=P_cost_rule)
        
        def Objective_rule(model):
            return sum(sum(sum(model.P_cost[n,i,t] for n in model.Gens) for i in model.Buses) for t in model.Times)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    """
    Objective Function - Equation (1)
     - Minimize loss
    """
    if 1 == 0:    
        # Equation (1)
        def Objective_rule(model):
            return sum(model.P_line_loss[l,t] for l in model.Lines for t in model.Times)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model


# Line 상태를 반영할 수 있는 OPF 함수
def OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info,Time_info,Time_interval_info,Ta,Tp,DG_profile_df,Load_profile_df):

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
    model.TimeIntervals = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network conductivity matrix
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any) # Network susceptance matrix
    
    #Demand at each node in time t - Unit:PU
    def P_demand_rule(model,i,t):
        return sum(Load_profile_df.loc[d,'p_mw_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, model.Times, rule = P_demand_rule)
    def Q_demand_rule(model,i,t):
        return sum(Load_profile_df.loc[d,'q_mvar_'+str(t)]/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses,model.Times, rule = Q_demand_rule)
    
    #Distributed generation at each node in time t - Unit:PU
    def P_dg_rule(model,n,i,t):
        if i == Slackbus:
            return 0
        else:
            if Gen_info.loc[n,'bus']==i:
                return DG_profile_df.loc[n,'p_mw_'+str(t)]/base_MVA 
            else:
                return 0
    model.PDg = pyo.Expression(model.Gens, model.Buses, model.Times, rule = P_dg_rule)
    def Q_dg_rule(model,n,i,t):
        if i == Slackbus:
            return 0
        else:
            if Gen_info.loc[n,'bus']==i:
                return DG_profile_df.loc[n,'q_mvar_'+str(t)]/base_MVA
            else:
                return 0
    model.QDg = pyo.Expression(model.Gens, model.Buses,model.Times, rule = Q_dg_rule)
    
    #Fictious load at each bus in time t - UnitPU (22)(23)
    def Fictious_load_rule(model,i):
        if Slackbus != i:
            if i in list(Gen_info['bus']):
                return 1
            else:
                return 0
        else:
            return (-1)*(len(Gen_info['bus'])-1) # Slack bus 제외
    model.FLoad = pyo.Expression(model.Buses, rule = Fictious_load_rule)
    
    """
    Variables
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,model.Times,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,model.Times,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    #Generation variable
    model.PGen = pyo.Var(model.Gens, model.Buses, model.Times, within=pyo.Reals, initialize=0.0)
    model.QGen = pyo.Var(model.Gens, model.Buses, model.Times, within=pyo.Reals, initialize=0.0)
    
    #Line status variable
    model.Line_Status = pyo.Var(model.Lines, model.Times, within=pyo.Integers, bounds=(0, 1),initialize = 1)    
    
    #Transfer bus status variable
    model.Transfer_bus_Status = pyo.Var(model.Buses, model.TimeIntervals, within=pyo.Integers, bounds=(0, 1),initialize = 1)
    
    #Fictious flow
    model.Fictious_flow = pyo.Var(model.Lines, model.TimeIntervals, within=pyo.Reals, initialize = 0.0)        
    
    """
    Expressions - Flow - Equation
    """
    # Equation (11)
    def P_line_flow_sending_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i,t] * model.V_mag[i,t] + model.Bus_G[i,j] * model.V_mag[i,t]* model.V_mag[j,t] * pyo.cos(model.V_ang[i,t]-model.V_ang[j,t]) + model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[j,t] * pyo.sin(model.V_ang[i,t]-model.V_ang[j,t]))
    model.P_line_flow_sending = pyo.Expression(model.Lines,model.Times,rule = P_line_flow_sending_rule)
    
    # Equation (12)
    def Q_line_flow_sending_rule(model,l,t):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[i,t] + model.Bus_G[i,j] * model.V_mag[i,t]* model.V_mag[j,t] * pyo.sin(model.V_ang[i,t]-model.V_ang[j,t]) - model.Bus_B[i,j] * model.V_mag[i,t] * model.V_mag[j,t] * pyo.cos(model.V_ang[i,t]-model.V_ang[j,t]))
    model.Q_line_flow_sending = pyo.Expression(model.Lines,model.Times,rule = Q_line_flow_sending_rule)
    
    # Equation (13)
    def P_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return ((-1) * model.Bus_G[i, j] * model.V_mag[j, t] * model.V_mag[j, t]
                + model.Bus_G[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t])
                - model.Bus_B[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.sin(model.V_ang[i, t] - model.V_ang[j, t]))
    model.P_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=P_line_flow_receiving_rule)
    # Equation (14)
    def Q_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Bus_B[i, j] * model.V_mag[j, t] * model.V_mag[j, t]
                - model.Bus_G[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.sin(model.V_ang[i, t] - model.V_ang[j, t])
                - model.Bus_B[i, j] * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t]))
    model.Q_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=Q_line_flow_receiving_rule)

    # Equation (9)-left
    def S_line_flow_sending_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return model.P_line_flow_sending[l, t] ** 2 + model.Q_line_flow_sending[l, t] ** 2
    model.S_line_flow_sending = pyo.Expression(model.Lines, model.Times, rule=S_line_flow_sending_rule)

    # Equation (9)-right
    if 1==0:
        def S_line_flow_sending_con_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']

            if Line_info.loc[l, 'rate_MVA'] == 0:
                return pyo.Constraint.Skip
            else:
                return model.S_line_flow_sending[l, t] <= model.Line_Status[l,t] * (Line_info.loc[l, 'rate_MVA'] / base_MVA) ** 2
        model.S_line_flow_sending_con = pyo.Constraint(model.Lines, model.Times, rule=S_line_flow_sending_con_rule)
        
    # Equation (9)-left
    def S_line_flow_receiving_rule(model, l, t):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return model.P_line_flow_receiving[l, t] ** 2 + model.Q_line_flow_receiving[l, t] ** 2
    model.S_line_flow_receiving = pyo.Expression(model.Lines, model.Times, rule=S_line_flow_receiving_rule)
    # Equation (9)-right
    if 1==0:
        def S_line_flow_receiving_con_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']

            if Line_info.loc[l, 'rate_MVA'] == 0:
                return pyo.Constraint.Skip
            else:
                return model.S_line_flow_receiving[l, t] <= model.Line_Status[l,t] * (Line_info.loc[l, 'rate_MVA'] / base_MVA) ** 2
        model.S_line_flow_receiving_con = pyo.Constraint(model.Lines, model.Times, rule=S_line_flow_receiving_con_rule)
    
    """
    Line status
    1 Constraints
    """
    ##Line status constraint  (10)
    def Line_status_rule(model,l,t):
        return model.Line_Status[l,t] <= 1
    model.Line_Status_con = pyo.Constraint(model.Lines, model.Times, rule = Line_status_rule)

    """
    Radiality constraint - Transfer bus status
    4 Constraints
    """
    ##Transfer bus constraint 1 (15)
    def Transfer_rule1(model,l,ta):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        
        if Bus_info.loc[i,"Transfer_Bus"] == 1:
            return model.Line_Status[l,1+(ta-1)*Ta] <= model.Transfer_bus_Status[i,ta]
        else:
            return pyo.Constraint.Skip
                        
    model.Transfer_rule1_con = pyo.Constraint(model.Lines, model.TimeIntervals,rule = Transfer_rule1)
    
    ##Transfer bus constraint 2 (16)
    def Transfer_rule2(model,l,ta):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        
        if Bus_info.loc[j,"Transfer_Bus"] == 1:
            return model.Line_Status[l,1+(ta-1)*Ta] <= model.Transfer_bus_Status[j,ta]
        else:
            return pyo.Constraint.Skip
                        
    model.Transfer_rule2_con = pyo.Constraint(model.Lines, model.TimeIntervals,rule = Transfer_rule2)
    
    ##Transfer bus constraint 3 (17)
    def Transfer_rule3(model,i,ta):
        
        if Bus_info.loc[i,"Transfer_Bus"] == 1:
            return (
                sum(model.Line_Status[l,1+(ta-1)*Ta]
                for l in Line_info.index if Line_info.loc[l, "from_bus"] == i )
                + sum( model.Line_Status[l,1+(ta-1)*Ta]
                for l in Line_info.index if Line_info.loc[l, "to_bus"] == i)
                >= 2*model.Transfer_bus_Status[i,ta]
            )
        else:
            return pyo.Constraint.Skip
    model.Transfer_rule3_con = pyo.Constraint(model.Buses, model.TimeIntervals,rule = Transfer_rule3)
    
    ##Transfer bus constraint 4 (19)
    def Transfer_rule4(model,ta):
        return sum(model.Line_Status[l,1+(ta-1)*Ta] for l in model.Lines) == len(Bus_info) - 1 - sum(1 - model.Transfer_bus_Status[i,ta] for i in model.Buses if Bus_info.loc[i,"Transfer_Bus"] == 1)
    model.Transfer_rule4_con = pyo.Constraint(model.TimeIntervals,rule = Transfer_rule4)
    
    """
    Radiality constraint - Distributed generator, Reactive power generator
    4 Constraints
    """
    ##Fictious flow constraint 1 (20)
    def Fictious_flow_rule1(model,i,ta):
        return ( sum(model.Fictious_flow[l,ta] for l in model.Lines if Line_info.loc[l, "to_bus"] == i) - sum(model.Fictious_flow[l,ta] for l in model.Lines if Line_info.loc[l, "from_bus"] == i)) == model.FLoad[i]                
    model.Fictious_flow_rule1_con = pyo.Constraint(model.Buses, model.TimeIntervals,rule = Fictious_flow_rule1)
    
    ##Fictious flow constraint 2 (24)
    def Fictious_flow_rule2(model,l,ta):
        return (-1)*(len(Gen_info)-1)*model.Line_Status[l,1+(ta-1)*Ta] <= model.Fictious_flow[l,ta]
    model.Fictious_flow_rule2_con = pyo.Constraint(model.Lines, model.TimeIntervals,rule = Fictious_flow_rule2)

    def Fictious_flow_rule3(model,l,ta):
        return model.Fictious_flow[l,ta] <= (len(Gen_info)-1)*model.Line_Status[l,1+(ta-1)*Ta]
    model.Fictious_flow_rule3_con = pyo.Constraint(model.Lines, model.TimeIntervals,rule = Fictious_flow_rule3)
    
    """
    Radiality constraint - Maintaining line status
    1 Constraints
    """
    # Equation (25)
    def Line_status_time_interval_rule(model, l, ta):
       return sum(model.Line_Status[l, h] for h in range(1 + (ta-1)*Ta, ta*Ta + 1)) == model.Line_Status[l, 1 + (ta-1)*Ta] * Ta
    model.Line_Status_time_interval_con = pyo.Constraint(model.Lines, model.TimeIntervals, rule=Line_status_time_interval_rule)
    
    # Equation (26) - use if Equation (25) not works
    # def Line_status_time_interval_rule2(model, l, t):
    #    for ta in range(1, Tp+1):
    #        if (t >= 1 +(ta-1)*Ta) and (t<ta*Ta):
    #            return model.Line_Status[l, t+1] == model.Line_Status[l, t]
    #        else:
    #            return pyo.Constraint.Skip
    # model.Line_Status_time_interval_con2 = pyo.Constraint(model.Lines, model.Time, rule=Line_status_time_interval_rule2)
    
    
    """
    Constraints - Load Balance (Generation - Demand) - Equation
    2 Constraints
    """
    # Power injection at each node (2), (3)
    def P_bal_rule(model, i, t):
        return (
            sum(model.PGen[n, i, t] for n in model.Gens)
            - model.PDem[i, t]
            == (
                sum(  
                    model.Line_Status[l,t] * model.P_line_flow_sending[l, t]
                    for l in model.Lines
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum(
                    model.Line_Status[l,t] * model.P_line_flow_receiving[l, t]
                    for l in model.Lines
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
                    model.Line_Status[l,t] * model.Q_line_flow_sending[l, t]
                    for l in model.Lines
                    if Line_info.loc[l, "from_bus"] == i
                )
                + sum(
                    model.Line_Status[l,t] * model.Q_line_flow_receiving[l, t]
                    for l in model.Lines
                    if Line_info.loc[l, "to_bus"] == i
                )
                - sum(model.Line_Status[l,t] * Line_info.loc[l,'b_pu']/2  for l in model.Lines if Line_info.loc[l, "from_bus"] == i) * model.V_mag[i,t] * model.V_mag[i,t]
                - sum(model.Line_Status[l,t] * Line_info.loc[l,'b_pu']/2  for l in model.Lines if Line_info.loc[l, "to_bus"] == i) * model.V_mag[i,t] * model.V_mag[i,t]
            )
        )
    model.Q_bal_con = pyo.Constraint(model.Buses, model.Times, rule=Q_bal_rule)
    
    """
    Constraints - Power and voltage - Equation 
    - Power
    4 Constraints
    2 Expressions - Convert PU to MW,MVar
    
    - Voltage
    2 Constraints
    2 Expression - Voltage magnitude [kV], angle [deg]
    """
    
    # Equation (4) - Min, Unit: [PU]
    def P_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return Gen_info.loc[n, 'min_p_mw'] / base_MVA <= model.PGen[n, i, t]
            else:
                return 0 <= model.PGen[n, i, t]
        else:
            return 0 <= model.PGen[n, i, t]
    model.P_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_min_rule)

    # Equation (4) - Max, Unit: [PU]
    def P_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return model.PGen[n, i, t] <= Gen_info.loc[n, 'max_p_mw'] / base_MVA
            else:
                return model.PGen[n, i, t] <= model.PDg[n,i,t]
        else:
            return model.PGen[n, i, t] <= 0
    model.P_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=P_gen_max_rule)

    # Equation (5) - Min, Unit: [PU]
    def Q_gen_min_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return Gen_info.loc[n, 'min_q_mvar'] / base_MVA <= model.QGen[n, i, t]
            else:
                # return model.QDg[n,i,t] * (-1) <= model.QGen[n, i, t]
                return 0 <= model.QGen[n, i, t]
        else:
            return 0 <= model.QGen[n, i, t]
    model.Q_gen_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_min_rule)

    # Equation (5) - Max, Unit: [PU]
    def Q_gen_max_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return model.QGen[n, i, t] <= Gen_info.loc[n, 'max_q_mvar'] / base_MVA
            else:
                return model.QGen[n, i, t] <= model.QDg[n,i,t]
        else:
            return model.QGen[n, i, t] <= 0
    model.Q_gen_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=Q_gen_max_rule)

    """
    Expressions - Curtailment
    """

    # Expression - PV curtailment - Unit:MW
    def PV_curtailment_rule(model, n, i, t):
        if Gen_info.loc[n, 'bus'] == i:
            if i == Slackbus:
                return 0
            else:
                return (model.PDg[n, i, t] - model.PGen[n, i, t]) * base_MVA
        else:
            return 0
    model.PV_curtailment_value = pyo.Expression(model.Gens, model.Buses, model.Times, rule=PV_curtailment_rule)

    """
    Expression - Net load
    """

    # Expression - Net load - Unit: MW
    def Net_load_rule(model, i, t):
        return (model.PDem[i, t] - sum(model.PGen[n, i, t] for n in model.Gens if Gen_info.loc[n, 'bus'] == i)) * base_MVA
    
    model.Net_load = pyo.Expression(model.Buses, model.Times, rule=Net_load_rule)

    """
    Expression - Net Power Flow
    """




    # """
    # Constraints - PV Curtailments
    # 3 Constraints
    # """

    # def PV_P_curtailment_rule(model, n, i, t):
    #     if Gen_info.loc[n, 'bus'] == i and n in pv_curtailment_df['Gen number'].values:
    #             pv_p_max_mw = pv_curtailment_df.loc[pv_curtailment_df['Gen number'] == n, 'PV_max[MW]'].values[0]
    #             pv_p_max_pu = pv_p_max_mw / base_MVA
    #             return model.PGen[n, i, t] <= pv_p_max_pu
    #     else:
    #         return pyo.Constraint.Skip
    
    # model.PV_P_curtailment_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=PV_P_curtailment_rule)

    # def PV_Q_curtailment_max_rule(model, n, i, t):
    #     if Gen_info.loc[n, 'bus'] == i and n in pv_curtailment_df['Gen number'].values:
    #         pv_p_max_mw = pv_curtailment_df.loc[pv_curtailment_df['Gen number'] == n, 'PV_max[MW]'].values[0]
    #         power_factor = 0.95
    #         tan_angle = ((1 - power_factor ** 2) / power_factor) ** 0.5
    #         pv_q_max_pu = pv_p_max_mw * tan_angle / base_MVA
    #         return model.QGen[n, i, t] <= pv_q_max_pu
    #     else:
    #         return pyo.Constraint.Skip

    # model.PV_Q_curtailment_max_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=PV_Q_curtailment_max_rule)

    # def PV_Q_curtailment_min_rule(model, n, i, t):
    #     if Gen_info.loc[n, 'bus'] == i and n in pv_curtailment_df['Gen number'].values:
    #         pv_p_max_mw = pv_curtailment_df.loc[pv_curtailment_df['Gen number'] == n, 'PV_max[MW]'].values[0]
    #         power_factor = 0.95
    #         tan_angle = ((1 - power_factor ** 2) / power_factor) ** 0.5
    #         pv_q_max_pu = pv_p_max_mw * tan_angle / base_MVA
    #         return (-1) * pv_q_max_pu <= model.QGen[n, i, t]
    #     else:
    #         return pyo.Constraint.Skip

    # model.PV_Q_curtailment_min_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=PV_Q_curtailment_min_rule)

    # def PV_S_curtailment_rule(model, n, i, t):

    #     if Gen_info.loc[n, 'bus'] == i and n in pv_curtailment_df['Gen number'].values:
    #         pv_max_mw = pv_curtailment_df[pv_curtailment_df['Gen number'] == n]['PV_max[MW]'].values[0]
    #         pv_p_max_pu = pv_max_mw / base_MVA

    #         power_factor = 0.95
    #         tan_angle = ((1 - power_factor ** 2) / power_factor) ** 0.5
    #         pv_q_max_mw = pv_max_mw * tan_angle
    #         pv_q_max_pu = pv_q_max_mw / base_MVA
    #         pv_s_max_pu = (pv_p_max_pu ** 2 + pv_q_max_pu ** 2) ** 0.5
            
    #         pdg_squared = model.PGen[n, i, t] ** 2
    #         qdg_squared = model.QGen[n, i, t] ** 2
    #         return (pdg_squared + qdg_squared) <= pv_s_max_pu ** 2
    #     else:
    #         return pyo.Constraint.Skip
    
    # model.PV_S_curtailment_con = pyo.Constraint(model.Gens, model.Buses, model.Times, rule=PV_S_curtailment_rule)


    # Expression - Active power expression at each node - Unit:MW
    def P_gen_MW_rule(model, n, i, t):
        return model.PGen[n, i, t] * base_MVA
    model.P_gen_MW = pyo.Expression(model.Gens, model.Buses, model.Times, rule=P_gen_MW_rule)

    # Expression - Reactive power expression at each node - Unit:MVar
    def Q_gen_MVar_rule(model, n, i, t):
        return model.QGen[n, i, t] * base_MVA
    model.Q_gen_MVar = pyo.Expression(model.Gens, model.Buses, model.Times, rule=Q_gen_MVar_rule)

    # Equation (6), Unit: [PU]
    def V_limits_rule(model, i,t):
        return (Bus_info['Vmin_pu'][i],model.V_mag[i,t],Bus_info['Vmax_pu'][i])
        # return (0.5,model.V_mag[i,t],1.5)
    model.V_limits_con = pyo.Constraint(model.Buses, model.Times, rule=V_limits_rule)
    
    def Slack_con_rule(model, i, t):
        if i == Slackbus:
            return model.V_ang[i,t] == 0
        else:
            return ((-1)*np.pi,model.V_ang[i,t],np.pi)
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
    if 1==0:
        # Equation (13)
        def I_line_re_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']
            return(
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
            return(
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
                return model.I_line_sq[l,t] <= model.Line_Status[l,t] * (9999999/base_current) ** 2                # If line limit info is existed, the limit replaces 999999.
            model.I_loading_con = pyo.Constraint(model.Lines, model.Times, rule = I_loading_con_rule)
    
    """
    Minimize Generation cost
    1 Expressions
    """
    if 1 == 1: 
        def P_cost_rule(model, n,i,t):
            if Gen_info.loc[n,'bus'] == i:
                try: # 2차항까지 있는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i,t]*base_MVA + Gen_info.loc[n,'c2'] * (model.PGen[n,i,t]*base_MVA)**2
                except: # 2차항이 없는 비용 곡선
                    return Gen_info.loc[n,'c0'] + Gen_info.loc[n,'c1'] * model.PGen[n,i,t]*base_MVA
            else:
                return 0.0
        model.P_cost = pyo.Expression(model.Gens, model.Buses, model.Times,rule=P_cost_rule)
        
        def Objective_rule(model):
            return sum(sum(sum(model.P_cost[n,i,t] for n in model.Gens) for i in model.Buses) for t in model.Times)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    """
    Objective Function - Equation (1)
     - Minimize loss
    """
    if 1 == 1: 
        # Equation (2)
        def P_line_loss_rule(model,l,t):
            i = Line_info.loc[l,'from_bus']
            j = Line_info.loc[l,'to_bus']
            return ( - model.Bus_G[i, j] * model.Line_Status[l,t] *
                    ( model.V_mag[i, t] ** 2 +  model.V_mag[j, t] ** 2
                     - 2 * model.V_mag[i, t] * model.V_mag[j, t] * pyo.cos(model.V_ang[i, t] - model.V_ang[j, t]))
                    ) * base_MVA
        model.P_line_loss = pyo.Expression(model.Lines, model.Times, rule = P_line_loss_rule)
        
    if 1 == 0:
        def Q_line_loss_rule(model, l, t):
            i = Line_info.loc[l, 'from_bus']
            j = Line_info.loc[l, 'to_bus']
            return (model.Q_line_flow_sending[l, t] + model.Q_line_flow_receiving[l, t]) * base_MVA
        model.Q_line_loss = pyo.Expression(model.Lines, model.Times, rule=Q_line_loss_rule)
    
    if 1 == 0:    
        # Equation (1)
        def Objective_rule(model):
            return sum(model.P_line_loss[l,t] for l in model.Lines for t in model.Times)
        model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model