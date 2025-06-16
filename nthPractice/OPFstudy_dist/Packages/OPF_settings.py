"""
    Create OPF model without switch
"""

def OPF_model_without_switch(np, pyo, base_MVA, Slackbus, Bus_info, Line_info, Load_info, Gen_info):
   
    model = pyo.AbstractModel()

    """
    set params
    """

    model.Buses = pyo.Set(dimen=1)
    model.Lines = pyo.Set(dimen=1)
    model.Gens = pyo.Set(dimen=1)
    model.Loads = pyo.Set(dimen=1)

    model.Bus_G = pyo.Param(model.Buses,model.Buses,within=pyo.Any)
    model.Bus_B = pyo.Param(model.Buses,model.Buses,within=pyo.Any)

    def P_demand_rule(model,i):
        return sum(Load_info.loc[d,'p_mw'] for d in model.Loads if Load_info.loc[d,'bus'] == i)
    model.PDem = pyo.Expression(model.Buses, rule=P_demand_rule)
    def Q_demand_rule(model,i):
        return sum(Load_info.loc[d,'q_mvar'] for d in model.Loads if Load_info.loc[d,'bus'] == i)
    model.QDem = pyo.Expression(model.Buses, rule=Q_demand_rule)

    """
    set vars
    """

    model.V_mag = pyo.Var(model.Buses, within=pyo.NonNegativeReals, initialize=1.0)
    model.V_ang = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)

    def P_gen_init_rule(model, i):
        return (sum(Gen_info.loc[n, 'p_mw'] for n in model.Gens if Gen_info.loc[n, 'bus'] == i))
    
    model.PGen = pyo.Var(model.Buses, within=pyo.Reals, initialize=P_gen_init_rule)
    model.Qgen = pyo.Var(model.Buses, within=pyo.Reals, initialize=0.0)

    """
    set Equations
    1. Power flow equations
    """
    def P_line_flow_sending_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return ((-1) * model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i] - model.V_ang[j]) + model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i] - model.V_ang[j])) *base_MVA
    model.P_line_flow_sending = pyo.Expression(model.Lines, rule = P_line_flow_sending_rule)

    def Q_line_flow_sending_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return(model.Bus_B[i,j] * model.V_mag[i]* model.V_mag[i] + model.Bus_G[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.sin(model.V_ang[i] - model.V_ang[j]) - model.Bus_B[i,j] * model.V_mag[i] * model.V_mag[j] * pyo.cos(model.V_ang[i] - model.V_ang[j])) *base_MVA
    model.Q_line_flow_sending = pyo.Expression(model.Lines, rule = Q_line_flow_sending_rule)

    def P_line_flow_receiving_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return ((-1) * model.Bus_G[j,i] * model.V_mag[j]* model.V_mag[j] + model.Bus_G[j,i] * model.V_mag[j]* model.V_mag[i] * pyo.cos(model.V_ang[j] - model.V_ang[i]) - model.Bus_B[j,i] * model.V_mag[j] *model.V_mag[i] * pyo.sin(model.V_ang[j] - model.V_ang[i])) * base_MVA
    model.P_line_flow_receiving = pyo.Expression(model.Lines, rule = P_line_flow_receiving_rule)

    def Q_line_flow_receiving_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Bus_B[j,i] * model.V_mag[j]* model.V_mag[j] + model.Bus_G[j,i] * model.V_mag[j] * model.V_mag[i] * pyo.sin(model.V_ang[j] - model.V_ang[i]) + model.Bus_B[j,i] * model.V_mag[j] * model.V_mag[i] * pyo.cos(model.V_ang[j] - model.V_ang[i])) * base_MVA
    model.Q_line_flow_receiving = pyo.Expression(model.Lines, rule = Q_line_flow_receiving_rule)

    def P_line_loss_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.P_line_flow_sending[l] + model.P_line_flow_receiving[l])
    model.P_line_loss = pyo.Expression(model.Lines, rule = P_line_loss_rule)

    def Q_line_loss_rule(model, l):
        i = Line_info.loc[l, 'from_bus']
        j = Line_info.loc[l, 'to_bus']
        return (model.Q_line_flow_sending[l] + model.Q_line_flow_receiving[l])
    model.Q_line_loss = pyo.Expression(model.Lines, rule = Q_line_loss_rule)

    """
    set Expressions
    1. Voltage
    """

    def V_mag_kv_rule(model, i):
        return model.V_mag[i] * Bus_info.loc[i,'baseKV']
    model.V_mag_kv = pyo.Expression(model.Buses, rule=V_mag_kv_rule)

    def V_ang_deg_rule(model, i):
        return model.V_ang[i] * 180 / np.pi
    model.V_ang_deg = pyo.Expression(model.Buses, rule=V_ang_deg_rule)

    """
    set constraints
    1. load balance
    """

    def P_bal_rule(model, i):
        return model.PGen[i] - model.PDem[i] == (sum(model.P_line_flow_sending[l]/base_MVA for l in model.Lines if Line_info.loc[l,"from_bus"] == i) - sum(model.P_line_flow_receiving[l]/base_MVA for l in model.Lines if Line_info.loc[l,"to_bus"] == i))
    model.P_bal_con = pyo.Constraint(model.Buses, rule=P_bal_rule)
    def Q_bal_rule(model, i):
        return model.Qgen[i] - model.QDem[i] == (sum(model.Q_line_flow_sending[l]/base_MVA for l in model.Lines if Line_info.loc[l,"from_bus"] == i) - sum(model.Q_line_flow_receiving[l]/base_MVA for l in model.Lines if Line_info.loc[l,"to_bus"] == i)) 
    model.Q_bal_con = pyo.Constraint(model.Buses, rule=Q_bal_rule)

    """
    set constraints
    2. Power
    """

    def P_gen_min_rule(model, i):
        return (sum(Gen_info.loc[n, 'min_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n, 'bus'] == i)) <= model.PGen[i]
    model.P_gen_min_con = pyo.Constraint(model.Buses, rule=P_gen_min_rule)

    def P_gen_max_rule(model, i):
        return model.PGen[i] <= (sum(Gen_info.loc[n, 'max_p_mw']/base_MVA for n in model.Gens if Gen_info.loc[n, 'bus'] == i))
    model.P_gen_max_con = pyo.Constraint(model.Buses, rule=P_gen_max_rule)

    def Q_gen_min_rule(model, i):
        return (sum(Gen_info.loc[n, 'min_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n, 'bus'] ==i )) <= model.Qgen[i]
    model.Q_gen_min_con = pyo.Constraint(model.Buses, rule=Q_gen_min_rule)

    def Q_gen_max_rule(model, i):
        return model.Qgen[i] <= (sum(Gen_info.loc[n, 'max_q_mvar']/base_MVA for n in model.Gens if Gen_info.loc[n, 'bus'] ==i))
    model.Q_gen_max_con =pyo.Constraint(model.Buses, rule = Q_gen_max_rule)

    """
    set constraints
    2. Voltage
    """

    def V_limits_rule(model, i):
        return (Bus_info['Vmin_pu'][i],model.V_mag[i], Bus_info['Vmax_pu'][i])
    model.V_limits_con = pyo.Constraint(model.Buses, rule=V_limits_rule)

    """
    set constraints
    3. Slack
    """   

    def Slack_con_rule(model, i):
        if i == Slackbus:
            return model.V_ang[i] == 0.0
        else:
            return pyo.Constraint.Skip
    model.Slack_con = pyo.Constraint(model.Buses, rule=Slack_con_rule)


    """
    set constraints
    4. Current
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
    
    def I_loading_con_rule(model,l):
        base_current = base_MVA /Bus_info['baseKV'][Line_info.loc[l,"from_bus"]] / np.sqrt(3)
        return model.I_line_sq[l] <= (9999999/base_current) ** 2
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