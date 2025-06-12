"""
OPF model creator for balanced system
- PU를 적용하여 OPF를 풀 수 있는 시스템에 적용 가능
- Unbalanced system은 PU를 적용하기에 까다로울 것임

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
def OPF_model_creator_without_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info):

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
    def P_gen_ini_rule(model,i):
        return  (sum(Gen_info.loc[n,'p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i))
    
    model.PGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=P_gen_ini_rule)
    model.QGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    
    
    """
    Expressions - Flow - Equation (2) - (4), (7)
    """
    # Equation (3)
    def P_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) + model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.P_line_flow_sending = pyo.Expression(model.Lines,rule = P_line_flow_sending_rule)
    
    # Equation (7)
    def Q_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.Q_line_flow_sending = pyo.Expression(model.Lines,rule = Q_line_flow_sending_rule)
    
    # Equation (4)
    def P_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[j] * model.V_mag[j] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.P_line_flow_receiving = pyo.Expression(model.Lines,rule = P_line_flow_receiving_rule)    
    
    def Q_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[j] * model.V_mag[j] - model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.Q_line_flow_receiving = pyo.Expression(model.Lines,rule = Q_line_flow_receiving_rule)
    
    # Equation (2)
    def P_line_loss_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.P_line_flow_sending[l] + model.P_line_flow_receiving[l])
    model.P_line_loss = pyo.Expression(model.Lines,rule = P_line_loss_rule)
    
    def Q_line_loss_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Q_line_flow_sending[l] + model.Q_line_flow_receiving[l])
    model.Q_line_loss = pyo.Expression(model.Lines,rule = Q_line_loss_rule)
    
    """
    Constraints - Load Balance (Generation - Demand) - Equation (5)-(6)
    2 Constraints
    """
    # Power injection at each node
    def P_bal_rule(model, i):
        return model.PGen[i] - model.PDem[i] == ( sum (model.P_line_flow_sending[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.P_line_flow_receiving[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) )
    model.P_bal_con = pyo.Constraint(model.Buses,rule=P_bal_rule)
    
    def Q_bal_rule(model, i):
        return model.QGen[i] - model.QDem[i] == ( sum (model.Q_line_flow_sending[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.Q_line_flow_receiving[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) )
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
    def P_gen_min_rule(model,i):
        return  (sum(Gen_info.loc[n,'min_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) <= model.PGen[i]
    model.P_gen_min_con = pyo.Constraint(model.Buses, rule =P_gen_min_rule)
    
    # Equation (8) - Max, Unit: [PU]
    def P_gen_max_rule(model,i):
        return  model.PGen[i] <= (sum(Gen_info.loc[n,'max_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) 
    model.P_gen_max_con = pyo.Constraint(model.Buses, rule =P_gen_max_rule)
    
    # Equation (9) - Min, Unit: [PU]
    def Q_gen_min_rule(model,i):
        return (sum(Gen_info.loc[n,'min_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) <= model.QGen[i]
    model.Q_gen_min_con = pyo.Constraint(model.Buses, rule =Q_gen_min_rule)
    
    # Equation (9) - Max, Unit: [PU]
    def Q_gen_max_rule(model,i):
        return model.QGen[i] <= (sum(Gen_info.loc[n,'max_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i))
    model.Q_gen_max_con = pyo.Constraint(model.Buses, rule =Q_gen_max_rule)
    
    # Expression - Active power expression at each node - Unit:MW
    def P_gen_MW_rule(model,i):
        return  model.PGen[i] * base_MVA
    model.P_gen_MW = pyo.Expression(model.Buses, rule =P_gen_MW_rule)
    
    # Expression - Reactive power expression at each node - Unit:MW
    def Q_gen_MVar_rule(model,i):
        return  model.QGen[i] * base_MVA
    model.Q_gen_MVar = pyo.Expression(model.Buses, rule =Q_gen_MVar_rule)
    
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
    
    # Equation (12)
    def I_loading_con_rule(model,l):
        base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
        return model.I_line_sq[l] <= (9999999/base_current) ** 2                # If line limit info is existed, the limit replaces 999999.
    model.I_loading_con = pyo.Constraint(model.Lines,rule = I_loading_con_rule)
    
    """
    Objective Function - Equation (1)
     - Minimize loss
    """
    
    # Equation (1)
    def Objective_rule(model):
        return sum(model.P_line_loss[l] for l in model.Lines)
    model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model
   

# Line 상태를 반영할 수 있는 OPF 함수
def OPF_model_creator_with_switch(np,pyo,base_MVA,Slackbus,Bus_info,Line_info,Load_info,Gen_info):

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
    2 Variables - Voltage magnitude [PU], angle
    2 Constraints
    2 Expression - Voltage magnitude [kV], angle [deg]
    """
    # Voltage variable
    model.V_mag = pyo.Var(model.Buses,within=pyo.NonNegativeReals,initialize = 1)  # Voltage magnitude
    model.V_ang = pyo.Var(model.Buses,within=pyo.Reals,initialize = 0)  # Voltage angle
    
    # Voltage limit
    def V_limits_rule(model, i):
        return (Bus_info['Vmin_pu'][i],model.V_mag[i],Bus_info['Vmax_pu'][i])
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)
    
    # Slack constraint
    def Slack_con_rule(model, i):
        if i == Slackbus:
            return model.V_ang[i] == 0
        else:
            return pyo.Constraint.Skip
    model.Slack_con = pyo.Constraint(model.Buses, rule=Slack_con_rule)
    
    # Voltage expression - magnitude [kV]
    def V_mag_kv_rule(model, i):
        return model.V_mag[i] * Bus_info.loc[i,'baseKV']
    model.V_mag_kv = pyo.Expression(model.Buses, rule=V_mag_kv_rule)
    # Voltage expression - angle [deg]
    def V_ang_deg_rule(model, i):
        return model.V_ang[i] * 180 / np.pi
    model.V_ang_deg = pyo.Expression(model.Buses, rule=V_ang_deg_rule)
    
    """
    Power - Bus
    2 Variables - Active power and reactive power (Unit:PU)
    2 Constraints
    2 Expressions - Demand
    2 Expressions - Convert PU to MW,MVar
    """
    #Generation variable
    model.PGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    model.QGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)
    
    #Active power at each node - Unit:PU
    def P_gen_min_rule(model,i):
        return  (sum(Gen_info.loc[n,'min_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) <= model.PGen[i]
    model.P_gen_min_con = pyo.Constraint(model.Buses, rule =P_gen_min_rule)
    
    def P_gen_max_rule(model,i):
        return  model.PGen[i] <= (sum(Gen_info.loc[n,'max_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) 
    model.P_gen_max_con = pyo.Constraint(model.Buses, rule =P_gen_max_rule)
    
    #Reactive power at each node - Unit:PU
    def Q_gen_min_rule(model,i):
        return (sum(Gen_info.loc[n,'min_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i)) <= model.QGen[i]
    model.Q_gen_min_con = pyo.Constraint(model.Buses, rule =Q_gen_min_rule)
    
    def Q_gen_max_rule(model,i):
        return model.QGen[i] <= (sum(Gen_info.loc[n,'max_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n,'bus'] == i))
    model.Q_gen_max_con = pyo.Constraint(model.Buses, rule =Q_gen_max_rule)
    
    #Demand at each node - Unit:PU
    def P_demand_rule(model,i):
        return sum(Load_info.loc[d,'p_mw']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.PDem = pyo.Expression(model.Buses, rule = P_demand_rule)
    def Q_demand_rule(model,i):
        return sum(Load_info.loc[d,'q_mvar']/base_MVA for d in model.Loads if Load_info.loc[d,'bus']==i)
    model.QDem = pyo.Expression(model.Buses, rule = Q_demand_rule)
    
    #Active power expression at each node - Unit:MW
    def P_gen_MW_rule(model,i):
        return  model.PGen[i] * base_MVA
    model.P_gen_MW = pyo.Expression(model.Buses, rule =P_gen_MW_rule)
    
    def Q_gen_MVar_rule(model,i):
        return  model.QGen[i] * base_MVA
    model.Q_gen_MVar = pyo.Expression(model.Buses, rule =Q_gen_MVar_rule)
    
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
    
    #def Line_total_status_rule(model):
    #    return sum ( sum(model.Line_Status[i,j] for j in model.Buses) for i in model.Buses) <= (37) * 2 
    
    #model.Line_total_status_con = pyo.Constraint(rule = Line_total_status_rule)
        
    
    """
    Power - Line (Unit:MW, MVar)
    6 Expressions
    """
    ## Power flow in each line - Unit:MW
    # Sending power
    def P_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) + model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.P_line_flow_sending = pyo.Expression(model.Lines,rule = P_line_flow_sending_rule)
    
    def Q_line_flow_sending_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.Q_line_flow_sending = pyo.Expression(model.Lines,rule = Q_line_flow_sending_rule)
    
    # Receiving power
    def P_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[j] * model.V_mag[j] + model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.P_line_flow_receiving = pyo.Expression(model.Lines,rule = P_line_flow_receiving_rule)    
    
    def Q_line_flow_receiving_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.Bus_B[i,j] * model.V_mag[j] * model.V_mag[j] - model.Bus_G[i,j] * model.V_mag[i]* model.V_mag[j] * pyo.sin(model.V_ang[i]-model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i]-model.V_ang[j])) * base_MVA
    model.Q_line_flow_receiving = pyo.Expression(model.Lines,rule = Q_line_flow_receiving_rule)
    
    # Loss
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
    
    """
    Current
    1 Constraint
    5 Expressions
    """
    def I_line_re_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ( (-1)*model.Bus_G[i,j]*model.V_mag[i]*pyo.cos(model.V_ang[i]) + model.Bus_B[i,j]*model.V_mag[i]*pyo.sin(model.V_ang[i]) + model.Bus_G[i,j]*model.V_mag[j]*pyo.cos(model.V_ang[j]) - model.Bus_B[i,j]*model.V_mag[j]*pyo.sin(model.V_ang[j]) )
    model.I_line_re = pyo.Expression(model.Lines,rule = I_line_re_rule)
    
    def I_line_im_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return ((-1)*model.Bus_B[i,j]*model.V_mag[i]*pyo.cos(model.V_ang[i]) - model.Bus_G[i,j]*model.V_mag[i]*pyo.sin(model.V_ang[i]) + model.Bus_B[i,j]*model.V_mag[j]*pyo.cos(model.V_ang[j]) + model.Bus_G[i,j]*model.V_mag[j]*pyo.sin(model.V_ang[j]))
    model.I_line_im = pyo.Expression(model.Lines,rule = I_line_im_rule)
    
    def I_line_sq_rule(model,l):
        i = Line_info.loc[l,'from_bus']
        j = Line_info.loc[l,'to_bus']
        return (model.I_line_re[l] ** 2 + model.I_line_im[l] ** 2)
    model.I_line_sq = pyo.Expression(model.Lines,rule = I_line_sq_rule)
    
    def I_line_mag_rule(model,l):
        base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
        return pyo.sqrt(model.I_line_sq[l]) * base_current 
    model.I_line_mag = pyo.Expression(model.Lines,rule = I_line_mag_rule)
    
    """
    def I_loading_percent_rule(model,l):
        return model.I_line_mag[l] / Line_info.loc[l,"max_i_ka"] * 100
    model.I_loading_percent = pyo.Expression(model.Lines,rule = I_loading_percent_rule)
    
    def I_loading_con_rule(model,l):
        base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
        return model.I_line_sq[l] <= (Line_info.loc[l,"max_i_ka"]/base_current) ** 2
    model.I_loading_con = pyo.Constraint(model.Lines,rule = I_loading_con_rule)
    """
        
    """
    Balance (Generation - Demand)
    2 Constraints
    2 Expressions
    """
    # Power injection at each node
    def P_bal_rule(model, i):
        return model.PGen[i] - model.PDem[i] == ( sum (model.Line_Status[l]*model.P_line_flow_sending[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.Line_Status[l]*model.P_line_flow_receiving[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) )
    model.P_bal_con = pyo.Constraint(model.Buses,rule=P_bal_rule)
    
    def Q_bal_rule(model, i):
        return model.QGen[i] - model.QDem[i] == ( sum (model.Line_Status[l]*model.Q_line_flow_sending[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"from_bus"] == i ) ) + ( sum (model.Line_Status[l]*model.Q_line_flow_receiving[l]/base_MVA for l in Line_info.index if Line_info.loc[l,"to_bus"] == i ) )
    model.Q_bal_con = pyo.Constraint(model.Buses,rule=Q_bal_rule)
    
    # Power injection value (Gen - Demand)
    def P_inj_rule(model,i):
        return model.PGen[i] - model.PDem[i]
    model.P_inj = pyo.Expression(model.Buses,rule = P_inj_rule)
    def Q_inj_rule(model,i):
        return model.QGen[i] - model.QDem[i]
    model.Q_inj = pyo.Expression(model.Buses,rule = Q_inj_rule)
    
    """
    Generation cost
    1 Expressions
    """
    def P_cost_rule(model, i):
        return (sum(Gen_info.loc[n,'c0'] for n in model.Gens if Gen_info.loc[n,'bus'] == i)) + (sum(Gen_info.loc[n,'c1'] for n in model.Gens if Gen_info.loc[n,'bus'] == i)) * model.PGen[i]*base_MVA + (sum(Gen_info.loc[n,'c2'] for n in model.Gens if Gen_info.loc[n,'bus'] == i)) * (model.PGen[i]*base_MVA)**2
    model.P_cost = pyo.Expression(model.Buses,rule=P_cost_rule)
    
    """
    Objective Function
     - Minimize loss
    """
    def Objective_rule(model):
        return sum(model.P_line_loss[l] for l in model.Lines)
    model.obj = pyo.Objective(rule=Objective_rule,sense=pyo.minimize)
    
    return model